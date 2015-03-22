//a wrapper for the mallet maxent classifier

//mallet2.05 Maximum entropy classifier
//import cc.mallet.share.upenn.MaxEntShell;
//import cc.mallet.classify.*;
//import cc.mallet.types.*;

//old Maximum entropy classifier
import edu.umass.cs.mallet.share.upenn.MaxEntShell;
import edu.umass.cs.mallet.base.classify.*;
import edu.umass.cs.mallet.base.types.*;

import java.io.*;
import java.util.*;

public class MaxentClassifier{
    
    //CONSTANTS
    //MEMBER VARIABLES
    
    //The maximum entropy model used to classify nominals
    protected static Classifier model;


    //CONSTRUCTORS
    
    public MaxentClassifier(String modelFileName) {
	File modelFile = new File(modelFileName);
	try{
	    model=MaxEntShell.load(modelFile);
    	}catch(ClassNotFoundException e){
	    System.err.println(e);
            e.printStackTrace();
            System.exit(-1);
	}catch(IOException e){
	    System.err.println(e);
            e.printStackTrace();
            System.exit(-1);
	}
    }



    /*
     * Train the model on the given list of files.
     */
    public static void train(String goldfile, String modelFileName) {

	try{
	    String[][] trainMatrix=new String[3000000][];
	    String[] labelArray=new String[3000000];

	    int index = 0;
	    //the gaussian variance needs to be properly set by cross-validation
	    double var = 1.0;  

	    File modelFile = new File(modelFileName);

	    BufferedReader traindata = new BufferedReader(new InputStreamReader(new FileInputStream(goldfile)));
	    String line = "";
	    
	    long begin = System.currentTimeMillis();
	    while( (line = traindata.readLine()) != null){
		String[] parts = line.split("\\s+");
		labelArray[index] = parts[0];
		trainMatrix[index] = new String[parts.length - 1];
		for (int i = 1; i < parts.length; i++){
		    trainMatrix[index][i-1] = parts[i];
		}
		index++;
	    }

	    //copy the feature matrix to a new matrix
	    //to get rid of the empty rows
	    String[][] trainMatrix1 = new String[index][];
	    String[] labelArray1 = new String[index];
	    for (int i=0; i < index;i++){
		trainMatrix1[i] = trainMatrix[i];
		labelArray1[i] = labelArray[i];
	    }
	    //training, parameter estimation
	    model = MaxEntShell.train(trainMatrix1, labelArray1, var, modelFile);
	
	    double elapsed = ((double)(System.currentTimeMillis() - begin))/1000;
	    System.out.println("Elapsed Time: " + elapsed + "seconds");
	}catch(FileNotFoundException fn){
	    System.err.println(fn);
	}catch(UnsupportedEncodingException ue){
	    System.err.println(ue);
	}catch(IOException e){
	    System.err.println(e);
	    e.printStackTrace();
	    System.exit(-1);
	}  

    }



    // ============================================================
    //                       RUNNING
    // ============================================================

    
    public static void classify(String test_feature_vectors){

	try{
	    //input file
	    BufferedReader testdata = new BufferedReader(new InputStreamReader (new FileInputStream(test_feature_vectors)));
	    String line = "";
	    while((line = testdata.readLine()) != null){
		String [] feature_vector = line.split("\\s+");
		//System.out.println(line + ' ' + parts.length);
		//String[] feature_vector = new String[3];
		//for (int i = 0; i < parts.length; i++){
		//  feature_vector[i] = parts[i];
		//}
		Classification cl = MaxEntShell.classify(model, feature_vector);
		List outcome = new ArrayList();
		// Find the most probable outcome.
		String best_label = "";
		double best_p = -1.0;
		int best_i = 0;
		Labeling lab = cl.getLabeling();
		LabelAlphabet labels = lab.getLabelAlphabet();
		
		for (int c = 0; c < labels.size(); c++){
		    double p_i = lab.value(c);
		    if (p_i > best_p){
			best_i = c;
			best_p = p_i;
		    }
		}
	
		best_label = (String)labels.lookupObject(best_i);
		System.out.print(best_label + "\t" + best_p);

		//output the other labels as well
		for (int c = 0; c < labels.size(); c++){
		    double p_i = lab.value(c);
		    if (p_i != best_p){
			System.out.print( "\t" + (String)labels.lookupObject(c)  + "\t" + p_i);
		    }
		}
		System.out.println(" " + line);

	    }
	}catch(FileNotFoundException fn){
	    System.err.println(fn);
	}catch(UnsupportedEncodingException ue){
	    System.err.println(ue);
	}catch(IOException e){
	    System.err.println(e);
	    e.printStackTrace();
	    System.exit(-1);
	}  

    }



    // ============================================================
    //                       COMMAND-LINE INTERFACE
    // ============================================================

    
    public static void usage(int exitcode) {
        String s;
        s = "\nUsage:\n"+
            "    java MaxentClassifier -train -model=<modelfile> -gold=<train-data>\n"+
            "    java MaxentClassifier -classify -model=<modelfile> -input=<inputfile>\n\n"+
            "    <model> is the maximum entropy model file.\n\n";

        System.err.println(s);
        System.exit(exitcode);
    }

    public static void main(String args[]) {

        try {
	    // Argument processing.
	    String action = "";
	    String gold = null;
	    String input = null;
	    String modelFile = null;

	    for (int i=0; i<args.length; i++){
		    if (args[i].equals("-train")) action = "train";
		    else if (args[i].equals("-classify")) action = "classify";
		    else if (args[i].equals("-help")) usage(0);
		    else if (args[i].startsWith("-model="))
			modelFile = args[i].substring(7);
		    else if (args[i].startsWith("-input="))
			input = args[i].substring(7);
		    else if(args[i].startsWith("-gold="))
			gold = args[i].substring(6);
		    else {};
		}


	    // Expand directories
	    if (action.equals("train")){
		if (modelFile==null) usage(-1);
		System.err.println("Training model ("+modelFile+")");
		System.err.println("Training file (" + gold + ")");
		train(gold, modelFile);
	    }else if (action.equals("classify")){
		if (modelFile==null) usage(-1);
		System.err.println("Running model (" + modelFile);
		MaxentClassifier mc = new MaxentClassifier(modelFile);
		mc.classify(input);
	    }else{
		usage(-1);
	    }

	    System.exit(0);

	}catch (Exception e) {
            System.err.println(e);
            e.printStackTrace();
            System.exit(-1);
        }
    }
}    
