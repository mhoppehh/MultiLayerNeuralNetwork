import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class DynamicNeuralNetwork
{
	String file_name = "data_compact.csv";
	
	int NUMBER_INPUT_NEURONS = 784;
    int NUMBER_HIDDEN_NEURONS[] = 
    		new int[] {30, 30, 30};
    int NUMBER_OUTPUT_NEURONS = 10;
    int TOTAL_NUMBER_OF_LAYERS = 
    		NUMBER_INPUT_NEURONS +
    		NUMBER_HIDDEN_NEURONS.length +
    		NUMBER_OUTPUT_NEURONS;
    int TOTAL_DATA_SIZE = 50000;
    int TRAINING_PICS = 30000;
    int TEST = 10000;
    
    NeuralNetwork nn;
    
    public static void main (String [] args){
    	DynamicNeuralNetwork m = new DynamicNeuralNetwork();
    }
    
    public DynamicNeuralNetwork() {
    	startTimer();
    	
    	init_neural_network();
    	
    	Data train_data = load_training_data(file_name);
    	Data test_data 	= load_testing_data	(file_name);
    	
    }
    
    private Data load_training_data(String file_name) {
    	Data d;
    	Picture pics[] = new Picture[TOTAL_DATA_SIZE];
        System.out.println("Loading training data...");
        BufferedReader reader;
        File f;
        
        f = new File("data_compact.csv");
        try {
			reader = new BufferedReader(new FileReader(f));
	        for (int p = 0; p < TOTAL_DATA_SIZE; p++){
	            String answer = reader.readLine().replaceAll(" ", "");
	            pics[p] = new Picture(new double[784], new int[10]);
	            for (int y = 0; y < 10; y++){
	                int index = answer.indexOf(",");
	                String a = answer.substring(0,index);
	                pics[p].output[y] = Integer.parseInt(a);
	                answer = answer.substring(index + 1, answer.length());
	        }
	
            String string = reader.readLine().replaceAll(" ", "");
            for (int x = 0; x < 784; x++){
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
    
    private Data load_testing_data(String file_name) {
    	Data t;
    	Picture[] pics = new Picture[TEST];
        System.out.println("Loading testing data...");
        BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(new File(file_name)));
	        for (int p = 0; p < TEST; p++){
	            pics[p] = new Picture(new double[784], new int[10]);
	
	            String answer = reader.readLine().replaceAll(" ", "");
	            for (int y = 0; y < 10; y++){
	                int index = answer.indexOf(",");
	                String a = answer.substring(0,index);
	                pics[p].output[y] = Integer.parseInt(a);
	                answer = answer.substring(index + 1, answer.length());
	            }
	
	            String string = reader.readLine().replaceAll(" ", "");
	            for (int x = 0; x < 784; x++){
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
