#include <boost/asio.hpp>
#include <iostream>
#include <thread>
#include <memory>
#include <sstream>
#include <cstdlib>

using boost::asio::ip::tcp;
namespace asio = boost::asio;

class session : public std::enable_shared_from_this<session> {
public:
    session(tcp::socket socket)
        : socket_(std::move(socket)), socket_open_(true) {}

    void start() {
        read();
    }

    ~session() {
        close_socket();
    }

private:
    void close_socket() {
        if (socket_open_) {
            try {
                socket_.shutdown(boost::asio::socket_base::shutdown_send);
                socket_.close();
            } catch (...) {
                // Handle all exceptions here, potentially logging them
            }
            socket_open_ = false;
        }
    }

    void read() {
        auto self(shared_from_this());
        asio::async_read_until(socket_, asio::dynamic_buffer(data_), "\n",
            [this, self](boost::system::error_code ec, std::size_t length) {
                if (!ec) {
                    std::cout << "Request received, processing data." << std::endl;
                    process_data();
                } else {
                    std::cerr << "Error reading from socket: " << ec.message() << std::endl;
                    close_socket();
                }
            });
    }

    void process_data() {
        auto self = shared_from_this(); // Maintain a reference to keep the session alive
        std::istringstream iss(data_);
        std::string network, framework;
        iss >> network >> framework;

        std::thread([this, self, network, framework]() {
            std::string command = "sudo bash scripts/run-server.sh " + framework + " " + network;
            int result = system(command.c_str());
            if (result != 0) {
                std::cerr << "Command execution failed with return code " << result << std::endl;
            }

            std::string completion_message = "Computation completed.\n";
            asio::post(socket_.get_executor(), [this, self, completion_message]() {
                if (socket_open_) {
                    asio::async_write(socket_, asio::buffer(completion_message),
                        [this, self](boost::system::error_code ec, std::size_t) {
                            close_socket();
                        });
                }
            });
        }).detach(); // Consider alternatives to detaching
    }

    tcp::socket socket_;
    std::string data_;
    bool socket_open_;
};

class server {
public:
    server(asio::io_context& io_context, short port)
        : io_context_(io_context),
          acceptor_(io_context, tcp::endpoint(tcp::v4(), port)),
          socket_(io_context) {
        accept_connections();
    }

    void stop() {
        // Stop accepting new connections
        acceptor_.close();
        // Close all open sockets
        for (auto& socket : active_sockets_) {
            socket.close();
        }
        // Stop the io_context to allow run() to return
        io_context_.stop();
    }

private:
    void accept_connections() {
        acceptor_.async_accept(socket_,
            [this](boost::system::error_code ec) {
                if (!ec) {
                    std::cout << "New connection accepted." << std::endl;
                    std::make_shared<session>(std::move(socket_))->start();
                    accept_connections();
                } else if (acceptor_.is_open()) {
                    std::cerr << "Failed to accept new connection: " << ec.message() << std::endl;
                }
            });
    }

    asio::io_context& io_context_;
    tcp::acceptor acceptor_;
    tcp::socket socket_;
    std::vector<tcp::socket> active_sockets_;
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: server <port>\n";
            return 1;
        }

        asio::io_context io_context;
        server srv(io_context, std::stoi(argv[1]));

        std::thread server_thread([&io_context]() {
            io_context.run();
        });

        std::cout << "Server is running. Press enter to stop.\n";
        std::cin.get();

        srv.stop();
        server_thread.join();
        std::cout << "Server stopped." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}

