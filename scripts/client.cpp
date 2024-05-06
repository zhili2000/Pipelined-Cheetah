#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <cstdlib>

using boost::asio::ip::tcp;
namespace asio = boost::asio;

class client {
public:
    client(asio::io_context& io_context, const std::string& host, const std::string& port,
           const std::string& network, const std::string& framework)
        : io_context_(io_context),
          socket_(io_context),
          network_(network),
          framework_(framework) {
        // Resolve the host and port
        tcp::resolver resolver(io_context);
        endpoints_ = resolver.resolve(host, port);

        // Execute initial local command
        connect();
    }

private:
    void connect() {
        asio::async_connect(socket_, endpoints_,
            [this](boost::system::error_code ec, tcp::endpoint) {
                if (!ec) {
                    send_request();
                }
            });
    }

    void send_request() {
        std::string message = network_ + " " + framework_ + "\n";  // Ensure newline
	std::cout << "Sending message: " << message << std::flush;
        asio::async_write(socket_, asio::buffer(message),
            [this](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    std::cout << "Message sent, awaiting response." << std::endl;
        	    execute_connection_command();
                    read_response();
                }
            });
    }

    void read_response() {
        asio::async_read_until(socket_, asio::dynamic_buffer(response_), "\n",
            [this](boost::system::error_code ec, std::size_t length) {
                if (!ec) {
                    std::cout << "Server says: " << response_.substr(0, length) << std::endl;
                    execute_post_response_command(); // Execute after receiving the server's response
                }
            });
    }

    void execute_connection_command() {
        std::string command = "sudo bash scripts/run-client.sh " + framework_ + " " + network_;
        system(command.c_str());
    }

    void execute_post_response_command() {
        std::string command = "echo Computation done, check out the log file " + framework_ + "-" + network_ + "_client.log";
        system(command.c_str());
    }

    asio::io_context& io_context_;
    tcp::socket socket_;
    std::string response_;
    std::string network_;
    std::string framework_;
    tcp::resolver::results_type endpoints_;
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: client <host> <port> <framework> <network>\n";
            return 1;
        }
        asio::io_context io_context;
        std::string host = argv[1];
        std::string port = argv[2];
        std::string framework = argv[3];
        std::string network = argv[4];

        client c(io_context, host, port, network, framework);
        io_context.run();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}

