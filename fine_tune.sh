export CUDA_VISIBLE_DEVICES=-1  # Do not use GPU. Use 0 to use 1st GPU， etc.
export TF_CPP_MIN_LOG_LEVEL=1  # Log verbosity: no Info, yes Warning, yes Error.

# Directories
# Directory to the PPI_Bert folder
PROJECT_DIR=/home/manbish/projects/PPI_Bert
# Directory where all the pretrained models are in
PRETRAIN_DIR=$PROJECT_DIR/pretrained
# Directory of your data  
DATA_DIR=$PROJECT_DIR/AIMed
# Directory where the trained model will be saved
OUTPUT_DIR=$DATA_DIR/Trained_Model

# Model Config
# Instance_Model vs. Sentence_Model.
CLASSIFICATION_MODEL=Instance_Model  
# The pre-trained model you use. The sub-directory name under PRETRAIN_DIR
PRETRAINED_MODEL=uncased_L-2_H-128_A-2
# Whether or not all characters be converted to lower case. Use True if pretrained model is trained using uncased characters.
DO_LOWER_CASE=True   

# Training Parameters
TRAIN_BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=1.0
# Max length of tokens for training data. Sentences longer than this will be trimmed
MAX_SEQ_LENGTH=128
# Max length of tokens for testing data. Sentences longer than this will be trimmed.
MAX_PRED_SEQ_LENGTH=256

echo "***PARAMETERS***"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "PRETRAIN_DIR=$PRETRAIN_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

echo "CLASSIFICATION_MODEL=$CLASSIFICATION_MODEL"
echo "PRETRAINED_MODEL=$PRETRAINED_MODEL"
echo "DO_LOWER_CASE=$DO_LOWER_CASE"

echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "MAX_PRED_SEQ_LENGTH=$MAX_PRED_SEQ_LENGTH"

python run_classifier.py \
--model=$CLASSIFICATION_MODEL \
--do_train=true \
--do_eval=true \
--data_dir=$DATA_DIR \
--vocab_file=$PRETRAIN_DIR/$PRETRAINED_MODEL/vocab.txt \
--bert_config_file=$PRETRAIN_DIR/$PRETRAINED_MODEL/bert_config.json \
--init_checkpoint=$PRETRAIN_DIR/$PRETRAINED_MODEL/bert_model.ckpt \
--max_seq_length=$MAX_SEQ_LENGTH \
--max_pred_seq_length=$MAX_PRED_SEQ_LENGTH \
--do_lower_case=$DO_LOWER_CASE \
--train_batch_size=$TRAIN_BATCH_SIZE \
--learning_rate=$LEARNING_RATE \
--num_train_epochs=$NUM_TRAIN_EPOCHS \
--output_dir=$OUTPUT_DIR
