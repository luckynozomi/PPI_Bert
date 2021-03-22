export CUDA_VISIBLE_DEVICES=-1  # Do not use GPU. Use 0 to use 1st GPU， etc.
export TF_CPP_MIN_LOG_LEVEL=1  # Log verbosity: no Info, yes Warning, yes Error.

# Directories
# Directory to the PPI_Bert folder
PROJECT_DIR=/home/manbish/projects/PPI_Bert
# Directory where all the pretrained models are in
PRETRAIN_DIR=$PROJECT_DIR/pretrained
# Directory of your data  
DATA_DIR=$PROJECT_DIR/AIMed
# Directory where the trained model was saved.
# Files will be generated under this directory as well.
TRAINED_CLASSIFIER=$DATA_DIR/Trained_Model

# Model Config
# Instance_Model vs. Sentence_Model.
CLASSIFICATION_MODEL=Instance_Model
# The pre-trained model you use. The sub-directory name under PRETRAIN_DIR
PRETRAINED_MODEL=uncased_L-2_H-128_A-2
# Whether or not all characters be converted to lower case. Use True if pretrained model is trained using uncased characters.
DO_LOWER_CASE=True
# Max length of tokens for testing data. Sentences longer than this will be trimmed.
MAX_PRED_SEQ_LENGTH=256

python run_predict.py \
--model=$CLASSIFICATION_MODEL \
--data_dir=$DATA_DIR \
--vocab_file=$PRETRAIN_DIR/$PRETRAINED_MODEL/vocab.txt \
--bert_config_file=$PRETRAIN_DIR/$PRETRAINED_MODEL/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--do_lower_case=$DO_LOWER_CASE \
--max_seq_length=$MAX_PRED_SEQ_LENGTH \
--output_id="" \
--output_dir=$TRAINED_CLASSIFIER
