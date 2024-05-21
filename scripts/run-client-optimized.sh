. scripts/common.sh

if [ ! $# -eq 3 ]
then
  echo -e "${RED}Please specify the network to run.${NC}"
  echo "Usage: run-client.sh [cheetah|SCI_HE] [sqnet/resnet50] [num 1-100]"
else
  if ! contains "cheetah SCI_HE" $1; then
    echo -e "Usage: run-client.sh ${RED}[cheetah|SCI_HE]${NC} [sqnet|resnet50|densenet121] [num 1-100]"
	exit 1
  fi

  if ! contains "sqnet resnet50 densenet121" $2; then
    echo -e "Usage: run-client.sh [cheetah|SCI_HE] ${RED}[sqnet|resnet50|densenet121]${NC} [num 1-100]"
	exit 1
  fi

  if ! echo "$3" | awk '$0 ~ /^[0-9]+$/ && $0 >= 1 && $0 <= 100 { exit 0 } { exit 1 }'; then
    echo -e "Usage: run-server.sh [cheetah|SCI_HE] [sqnet|resnet50|densenet121]$ ${RED}[num 1-100]${NC}"
        exit 1
  fi
  # create a data/ to store the Ferret output
  mkdir -p data
  echo -e "Runing ${GREEN}build/bin/$2-$1${NC}, which might take a while...."
  # cat pretrained/$2_input_scale12_pred*.inp | build/bin/$2-$1 r=2 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS ip=$SERVER_IP p=$SERVER_PORT b=$3 1>$1-$2_client_$(date +%s%N).log

  if [ -e "batch.txt" ]; then
    rm batch.txt
  fi

  # Define the pattern
  pattern="pretrained/${2}_input_scale12_pred*.inp"

  # Find the first file that matches the pattern
  file=$(ls $pattern 2>/dev/null | head -n 1)

  # Generate batch.txt with the desired number of file names
  for ((i=0; i<$3; i++)); do
    echo "$file" >> batch.txt
  done

  cat batch.txt | build/bin/$2-$1 r=2 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS ip=$SERVER_IP p=$SERVER_PORT b=$3 #1>$1-$2_client_$(date +%s%N).log

  echo -e "Computation done, check out the log file ${GREEN}$1-$2_client_$(date +%s%N).log${NC}"
fi
