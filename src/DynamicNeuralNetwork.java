import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

public class DynamicNeuralNetwork
{
	String file_name = "data_compact.csv";
	
	int NUMBER_INPUT_NEURONS = 10;
    int NUMBER_HIDDEN_NEURONS[] = //At least 1
    		new int[] {10, 10, 10};
    int NUMBER_OUTPUT_NEURONS = 10;
    int TOTAL_NUMBER_OF_LAYERS = NUMBER_HIDDEN_NEURONS.length +	2;
    int TOTAL_DATA_SIZE 	= 50000;
    int TRAINING_PICS 		= 50000;
    int TEST 				= 10000;
    
    NeuralNetwork nn;
    
    public static void main (String [] args){
    	DynamicNeuralNetwork m = new DynamicNeuralNetwork();
    }
    
    public DynamicNeuralNetwork() {
    	
    	GUI g = new GUI();
    	
    	startTimer();
    	
    	init_neural_network();
    	
    	Data train_data = load_training_data(file_name, 10, 784);
    	Data test_data 	= load_testing_data	(file_name, 10, 784);
    	
    	randomize_weights_biases();
    	
    	int EPOCHS = 20;
        int M_BATCH_SIZE = 10;
        int LEARNING_RATE = 3;
    	
    }
    
    private void randomize_weights_biases() {
    	System.out.println("Assigning random values...");
    	for(int i = 0; i < NUMBER_HIDDEN_NEURONS.length; i++){
    		for (int j = 0; j < NUMBER_HIDDEN_NEURONS[i]; j++){
    			nn.hidden[i][j] = new SigmoidNeuron(0.0,new double[nn.sizes[i]], 0.0, new double[nn.sizes[i]]);
	            nn.hidden[i][j].bias = Math.random()*2.0 - 1.0;
	            nn.hidden[i][j].dBias = 0;
	            for (int k = 0; k < NUMBER_INPUT_NEURONS; k++){
	                nn.hidden[i][j].weights[k] = Math.random()*2.0 - 1.0;
	                nn.hidden[i][j].dWeights[j] = 0;
	            }
    			
    		}
    	}
    	for (int i = 0; i < NUMBER_OUTPUT_NEURONS; i++){
            nn.output[i] = new SigmoidNeuron(0.0,new double[nn.sizes[Math.max(0,nn.sizes.length - 2)]], 0.0, new double[nn.sizes[Math.max(0,nn.sizes.length - 2)]]);
            nn.output[i].bias = Math.random()*2.0 - 1.0;
            nn.output[i].dBias = 0;
            for (int j = 0; j < NUMBER_HIDDEN_NEURONS[NUMBER_HIDDEN_NEURONS.length - 1]; j++){
                nn.output[i].weights[j] = Math.random()*2.0 - 1.0;
                nn.output[i].dWeights[j] = 0;
            }
        }
    }
    
    private Data load_training_data(String file_name, int output_length, int input_length) {
    	Data d;
    	Picture pics[] = new Picture[TOTAL_DATA_SIZE];
        System.out.println("Loading training data...");
        BufferedReader reader;
        File f;
        
        f = new File("data_compact.csv");
        try {
			reader = new BufferedReader(new FileReader(f));
	        for (int p = 0; p < TOTAL_DATA_SIZE; p++){
	        	String answer = reader.readLine();
	            answer = answer.replaceAll(" ", "");
	            pics[p] = new Picture(new double[input_length], new int[output_length]);
	            for (int y = 0; y < output_length; y++){
	                int index = answer.indexOf(",");
	                String a = answer.substring(0,index);
	                pics[p].output[y] = Integer.parseInt(a);
	                answer = answer.substring(index + 1, answer.length());
	        }
	
            String string = reader.readLine().replaceAll(" ", "");
            for (int x = 0; x < input_length; x++){
                int index = string.indexOf(",");
                String pixel = string.substring(0,index);
                pics[p].pixels[x] = (double)(Integer.parseInt(pixel)) / 256;
                string = string.substring(index + 1, string.length());
            }
	
	        }
	        d = new Data(pics, pics.length);
	        reader.close();
	    	return d;
        } catch (Exception e) {
			e.printStackTrace();
			return null;
		}
    }
    
    private Data load_testing_data(String file_name, int output_length, int input_length) {
    	Data t;
    	Picture[] pics = new Picture[TEST];
        System.out.println("Loading testing data...");
        BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(new File(file_name)));
	        for (int p = 0; p < TEST; p++){
	            pics[p] = new Picture(new double[input_length], new int[output_length]);
	
	            String answer = reader.readLine().replaceAll(" ", "");
	            for (int y = 0; y < output_length; y++){
	                int index = answer.indexOf(",");
	                String a = answer.substring(0,index);
	                pics[p].output[y] = Integer.parseInt(a);
	                answer = answer.substring(index + 1, answer.length());
	            }
	
	            String string = reader.readLine().replaceAll(" ", "");
	            for (int x = 0; x < input_length; x++){
	                int index = string.indexOf(",");
	                String pixel = string.substring(0,index);
	                pics[p].pixels[x] = (double)(Integer.parseInt(pixel)) / 256;
	                string = string.substring(index + 1, string.length());
	            }
	        }
	        reader.close();
	        t = new Data(pics, pics.length);
	        
	        return t;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
    }
    
    /**
     * Initalizes the NeuralNetwork object based on the instance variables
     * above.
     */
    private void init_neural_network() {
    	int[] sizes = new int[TOTAL_NUMBER_OF_LAYERS];
    	sizes[0] 				= NUMBER_INPUT_NEURONS;
    	sizes[sizes.length - 1] = NUMBER_OUTPUT_NEURONS;
    	
    	SigmoidNeuron hidden[][] 	= new SigmoidNeuron[NUMBER_HIDDEN_NEURONS.length][0];
    	SigmoidNeuron output[] 		= new SigmoidNeuron[NUMBER_OUTPUT_NEURONS];
    	
    	for(int i = 0; i < hidden.length;i++) {
    		sizes[i + 1] 	= NUMBER_HIDDEN_NEURONS[i];
    		hidden[i] 		= new SigmoidNeuron[NUMBER_HIDDEN_NEURONS[i]];
    	}
    	
    	nn = new NeuralNetwork(TOTAL_NUMBER_OF_LAYERS, sizes, hidden, output);
    }
    
    /**
     * Timer classes that return the time between start and stop in ms.
     */
    private void startTimer() {
    	startTime = System.currentTimeMillis();
    }
    long startTime;
    private long stopTimer() {
    	return System.currentTimeMillis() - startTime;
    }
    
    /**
     * SigNeur holds the weights, bias, and deltas.
     * SigNeur does not need to be changed because it is a low level.
     */
    public class SigmoidNeuron{
        double bias;
        double weights[];
        double dBias;
        double dWeights[];
        public SigmoidNeuron(double bias, double weights[],
        double dBias, double dWeights[]){
            this.bias = bias;
            this.weights = weights;
            this.dBias = dBias;
            this.dWeights = dWeights;
        }
    }
    
    /**
     * Neural Network object that holdes all the nessisary SigNeurons.
     * 'hidden' is a 2D array of SigNeurs the first array holding the
     * layer of SigNeurs. The subarray being the layer of SigNeurs.
     */
    public class NeuralNetwork{
        int nLayers;
        int sizes[];
        SigmoidNeuron hidden[][];
        SigmoidNeuron output[];
        public NeuralNetwork(int nLayers, int sizes[], 
        SigmoidNeuron hidden[][], SigmoidNeuron output[]){
            this.nLayers = nLayers;
            this.sizes = sizes;
            this.hidden = hidden;
            this.output = output;
        }
    }
    
    
}
