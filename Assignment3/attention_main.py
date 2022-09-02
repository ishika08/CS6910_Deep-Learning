import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, GRU 
from tqdm import tqdm
import pdb
import config

from LoadData import ReadData
from CreateModel import BuildModelAttention
from Accuracy import CalculateAccuracy, BeamCalculateAccuracy, CalculateAccuracyAttention, BeamCalculateAccuracyAttention
from EncoderDecoderModels import InferenceModelsAttention

WANDB = 0

if WANDB:
	import wandb
	from wandb.keras import WandbCallback
	wandb.init(config={"batch_size": 64, "epochs": 10, "Cell_Type": "LSTM", "h_layer_size": 64, "emb_size": 64, "dropout": 0}, project="Deep-Learning-RNN")

	myconfig = wandb.config


############################## Language ##############################
Languages = {'Bengali': 'bn', 'Gujarati': 'gu', 'Hindi': 'hi', 'Kannada': 'kn', 'Malayalam': 'ml', 'Marathi': 'mr', 'Punjabi': 'pa', 'Sindhi': 'sd', 'Sinhala': 'si', 'Tamil': 'ta', 'Telugu': 'te', 'Urdu': 'ur'}
Language = Languages['Hindi']

############################## Path to the dataset ##############################
train_data_path = '/cbr/saish/Datasets/dakshina_dataset_v1.0/'+ Language +'/lexicons/'+ Language +'.translit.sampled.train.tsv'
val_data_path = '/cbr/saish/Datasets/dakshina_dataset_v1.0/'+ Language +'/lexicons/'+ Language +'.translit.sampled.dev.tsv'
test_data_path = '/cbr/saish/Datasets/dakshina_dataset_v1.0/'+ Language +'/lexicons/'+ Language +'.translit.sampled.test.tsv'

############ Main Program ############
def main(args):

	############################## Hyperparameters ##############################
	epochs = args.epochs                  
	optimizer = args.optimizer
	Cell_Type = args.Cell_Type
	l_rate = args.l_rate
	batch_size = args.batch_size               
	emb_size = args.embedding_size
	n_enc_dec_layers = args.n_enc_dec_layers
	hidden_layer_size = args.hidden_layer_size
	dropout = args.dropout
	beam_size = args.beam_size

	############################## Reading Train Data ##############################
	input_texts, target_texts, input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data = ReadData(train_data_path, "train")

	############################## Reading Validation Data ##############################
	val_input_texts, val_target_texts, _, _, _, _, _, _, _, _, val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = ReadData(val_data_path, "val", input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

	############################## Reading Test Data ##############################
	test_input_texts, test_target_texts, _, _, _, _, _, _, _, _, test_encoder_input_data, test_decoder_input_data, test_decoder_target_data = ReadData(test_data_path, "test", input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

	################################ Build A Model ################################
	model, decoder_dense = BuildModelAttention(Cell_Type, n_enc_dec_layers, hidden_layer_size, num_encoder_tokens, num_decoder_tokens, dropout, emb_size)

	################################ Train the Model ################################
	if optimizer == 'Adam':
		opt = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, decay=0.001)
	elif optimizer == 'Nadam':
		opt = Nadam(learning_rate=l_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

	model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])
	model.summary()

	### Fit the model ###
	if WANDB:
		wandb.run.name = "cell_" + Cell_Type + "_nedl_" + str(n_enc_dec_layers) + "_bms_" +  str(beam_size) + "_hls_" + str(hidden_layer_size) + "_embs_" + str(emb_size) + "_ep_" + str(epochs) + "_bs_" + str(batch_size) + "_op_" + optimizer + "_do_" + str(dropout) + "_lr_" + str(l_rate)
		model.fit(
		    [encoder_input_data, decoder_input_data],
		    decoder_target_data,
		    batch_size=batch_size,
		    epochs=epochs,
		    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
		    callbacks=[wandb.keras.WandbCallback()]
		)
	else:
		model.fit(
	    		[encoder_input_data, decoder_input_data],
	    		decoder_target_data,
	    		batch_size=batch_size,
	    		epochs=epochs,
	    		validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data)
			)

	################################ Save Model ################################
	model.save("s2s_attention")

	################################ Inference Models ################################
	encoder_model, decoder_model = InferenceModelsAttention(model, Cell_Type, hidden_layer_size, n_enc_dec_layers, decoder_dense)

	# Reverse-lookup token index to decode sequences back to something readable.
	reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
	reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

	# Count
	n_val_words = len(val_input_texts)
	n_train_words = len(input_texts)
	n_test_words = len(test_input_texts)
	

	################################ Calculate Accuracy ################################
	print('\n CALCULATING WORD-LEVEL ACCURACY!! \n')

	# Beam Search
	if beam_size > 1:
		print("Train Data")
		train_acc = BeamCalculateAccuracyAttention(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, n_train_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
		print("Validation Data")
		val_acc = BeamCalculateAccuracyAttention(val_encoder_input_data, encoder_model, decoder_model, val_input_texts, val_target_texts, n_val_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
		print("Test Data")
		test_acc = BeamCalculateAccuracyAttention(test_encoder_input_data, encoder_model, decoder_model, test_input_texts, test_target_texts, n_test_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
	# No Beam Search
	else:
		print("Train Data")
		train_acc = CalculateAccuracyAttention(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, n_train_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)
		print("Validation Data")
		val_acc = CalculateAccuracyAttention(val_encoder_input_data, encoder_model, decoder_model, val_input_texts, val_target_texts, n_val_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)
		print("Test Data")
		test_acc = CalculateAccuracyAttention(test_encoder_input_data, encoder_model, decoder_model, test_input_texts, test_target_texts, n_test_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)

	if WANDB:
		wandb.log({"word_level_acc": val_acc})

	print("Train Accuracy (exact string match): %f " % (train_acc))
	print("Validation Accuracy (exact string match): %f " % (val_acc))
	print("Test Accuracy (exact string match): %f " % (test_acc))
	

	#################### HeatMaps and Connectivity Plots ######################
	HeatMap = 0
	VisConn = 0
	if HeatMap:
		from DecodeText import DecodeSequenceAttention
		from InferAttention import PlotAttentionWeights
		from VisualizeConnectivity import visualize
		total_words = n_test_words
		for seq_index in tqdm(range(total_words), desc='HeatMap generation in Progress'):

			input_seq = test_encoder_input_data[seq_index : seq_index + 1]
			input_text = test_input_texts[seq_index]
			len_input = len(input_text)
			decoded_word, attention_weights = DecodeSequenceAttention(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)
			attention_mat, pred_char_seq = PlotAttentionWeights(input_seq, len_input, attention_weights, reverse_input_char_index, reverse_target_char_index, input_text)

			################### Connectivity Visualization ###################
			if VisConn:
				for input_timestep in range(len_input):
			  		visualize(attention_mat, result_list, input_timestep, input_text)


############################ Main Funtion ############################
if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
