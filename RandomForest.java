/** Random Forest Algorithm for CS170 Fall 2014
    by Evan Fossier
**/

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;
import java.util.Arrays;
import java.util.Collections;

public class RandomForest{
    public final int BAG_SIZE; //The number of training samples to pick randomly for each decision tree
    public final int NUM_TREES; //The number of decision trees to create
    public final int MAX_TREE_DEPTH; //The maximum depth allowed for a tree
    public final int VECTOR_SIZE = 57;
    // VECTOR_SIZE based on this specific Spam/NotSpam classifier where we have:
    // -    48 continuous real [0,100] attributes of type word_freq_WORD
    // -    6 continuous real [0,100] attributes of type char_freq_CHAR
    // -    1 continuous real [1,...] attribute of type capital_run_length_average
    // -    1 continuous integer [1,...] attribute of type capital_run_length_longest
    // -    1 continuous integer [1,...] attribute of type capital_run_length_total

    public int numTrainingPoints = 3601; //initial estimates taken from this specific training set
    public int numInputPoints = 500; //both numTrainingPoints and numInputPoints will be updated to correct values when parsing the respective files

    public ArrayList<double[]> trainingVectors; //Holds the training point vectors
    public ArrayList<Integer> trainingLabels; //Holds the training labels

    public DecisionTreeNode[] decisionTrees; //Array of the roots of the decision trees

    public ArrayList<double[]> inputVectors; //The vectors we will classify
    public ArrayList<Integer>  outputLabels; //The resulting labels


    public RandomForest(String trainingVectors, String trainingLabels, int numTrees, double bagMultiplier, int maxTreeDepth) throws FileNotFoundException, IOException{
        System.out.println("Initializing Random Forest with parameters: ");
        System.out.println("training vector file: "+ trainingVectors);
        System.out.println("training labels file: "+ trainingLabels);
        System.out.println("Number of trees to create: "+ numTrees);
        System.out.println("Bagging multiplier: "+bagMultiplier);
        System.out.println("Vector size: "+VECTOR_SIZE);
        System.out.println("Max Tree Depth: "+maxTreeDepth);
        System.out.println("");

        NUM_TREES = numTrees;
        MAX_TREE_DEPTH = maxTreeDepth;

        readTrainingVectors(trainingVectors);
        System.out.println("");
        readTrainingLabels(trainingLabels);

        BAG_SIZE = (int)(bagMultiplier * numTrainingPoints);
    }

    /**
     * Reads the training vectors from the input filename into the trainingVectorsReal and trainingVectorInteger array lists
     */
    private void readTrainingVectors(String trainingVectors) throws FileNotFoundException, IOException{
        BufferedReader br = new BufferedReader(new FileReader(trainingVectors));
        System.out.println("Reading in training vectors from file "+trainingVectors);

        this.trainingVectors = new ArrayList<double[]>(numTrainingPoints);

        String line = "";
        String splitOn = ",";
        int numPoints = 0;
        while((line = br.readLine()) != null){
            String[] vector = line.split(splitOn);

            double[] realPortion = new double[VECTOR_SIZE];
            for(int i=0; i<VECTOR_SIZE; i++){
                realPortion[i] = Double.parseDouble(vector[i]);
            }
            this.trainingVectors.add(realPortion);

            numPoints++;
        }
        br.close();

        System.out.println("Total training vectors processed: "+numPoints);
        numTrainingPoints = numPoints;
    }

    /**
     * Reads the training labels from inputted file name into trainingLabels array
     */
    private void readTrainingLabels(String trainingLabels) throws FileNotFoundException, IOException{
        BufferedReader br = new BufferedReader(new FileReader(trainingLabels));
        System.out.println("Reading in training labels from file "+trainingLabels);

        this.trainingLabels = new ArrayList<Integer>(numTrainingPoints);
        String vector = "";
        String splitOn = ",";
        int numPoints = 0;
        while((vector = br.readLine()) != null){
            this.trainingLabels.add(Integer.parseInt(vector));
            numPoints++;
        }

        br.close();
        System.out.println("Total training labels processed: "+numPoints);
    }

    /**
     * Creates NUM_TREES decision trees from the input data
     */
    public void train(){
        System.out.println("Starting training, creating "+NUM_TREES+" decision trees");
        this.decisionTrees = new DecisionTreeNode[NUM_TREES];

        for(int i=0; i<NUM_TREES; i++){
            System.out.println("Creating tree "+ i);
            // BAGGING: pick BAG_SIZE elements with replacement
            int[] trainingSubset = sampleWithReplacement(BAG_SIZE, numTrainingPoints);

            this.decisionTrees[i] = new DecisionTreeNode(trainingSubset, 0, 0);
        }

    }

    /**
     * Reads in the input vectors that will be classified
     * @param filename filename (csv) containing the vectors
     */
    public void readInputVectors(String filename) throws FileNotFoundException, IOException{
        BufferedReader br = new BufferedReader(new FileReader(filename));
        System.out.println("Reading input vectors from file "+filename);

        String line = "";
        int numPoints = 0;

        this.inputVectors = new ArrayList<double[]>(numInputPoints);

        while((line = br.readLine()) != null){
            String[] fields = line.split(",");

            assert fields.length == VECTOR_SIZE;

            double[] v = new double[VECTOR_SIZE];
            for(int i=0; i<VECTOR_SIZE; i++){
                v[i] = Double.parseDouble(fields[i]);
            }

            this.inputVectors.add(v);
            numPoints++;
        }

        System.out.println("Total input vectors processed: "+numPoints);
        numInputPoints = numPoints;

        br.close();
    }

    /**
     * Classifies the input vectors in the arraylist inputVectors and outputs the labels to outputLabels arraylist
     * Runs each input vector on every decision tree and takes the majority label as our prediction
     */
    public void classifyInput(){
        if(this.inputVectors == null || this.inputVectors.size() == 0){
            System.out.println("Error: need to read the input vectors first!");
            return;
        }
        System.out.println("Starting classify");
        System.out.println("Number of input points: "+this.inputVectors.size());
        System.out.println("Number of decision trees: "+this.decisionTrees.length);

        this.outputLabels = new ArrayList<Integer>(numInputPoints);

        // Walkthrough the input points
        for(int i=0; i<numInputPoints; i++){
            // System.out.println("Classifying point "+i);
            double[] vector = this.inputVectors.get(i);

            // If only one tree then the label is just the result of that one tree
            if(NUM_TREES == 1){
                this.outputLabels.add(traverse(this.decisionTrees[0], vector));
            }else{
                int[] labels = new int[NUM_TREES];
                int zeroCount = 0;
                for(int j=0; j<NUM_TREES; j++){
                   if(traverse(this.decisionTrees[j], vector) == 0){
                        zeroCount++;
                   }
                }
                // Resolve ties in favor of 0 (NOT SPAM) 
                if(zeroCount >= NUM_TREES/2){
                    this.outputLabels.add(0);
                }
                else{
                    this.outputLabels.add(1);
                }
            }
        }
    }

    /**
     * Traverses the given decision tree with the input vector and returns the label of the leaf node we arrive at. 
     * @param  root     the root of the tree to traverse
     * @param  vector   The vector to classify
     * @return          the label of the leaf node we arrive at.
     */
    private int traverse(DecisionTreeNode root, double[] vector){
        DecisionTreeNode cur = root;
        // Keep going until we get to a leaf
        while(cur != null && !cur.isLeaf){
            double val = vector[cur.featureNum];
            if(val <= cur.boundary){
                cur = cur.left;
            }else{
                cur = cur.right;
            }
        }

        if(cur == null){
            System.out.println("Error: got to a null child somehow, the tree was not constructed properly");
            return -1;
        }

        return cur.label;
    }

    /**
     * Checks the output of the classification with the known correct output
     * @param  correctOutput         Filename containing the correct output labels
     * @param  printErrors           true if you want to print each error, false to just get overall accuracy rate
     * @throws FileNotFoundException 
     * @throws IOException           
     */
    public void checkOutput(String correctOutput, boolean printErrors) throws FileNotFoundException, IOException{
        BufferedReader br = new BufferedReader(new FileReader(correctOutput));
        System.out.println("Checking output classification versus labels in "+correctOutput);

        if(this.outputLabels == null || this.outputLabels.size() == 0){
            System.out.println("Error: need to classify the input points before checking output");
            return;
        }

        int pointNum = 0;
        int numErrors = 0;
        String line = "";
        while((line = br.readLine()) != null){
            int correctLabel = Integer.parseInt(line);
            if(this.outputLabels.get(pointNum) != correctLabel){
                if(printErrors)
                    System.out.println("Error on point "+pointNum+": was "+this.outputLabels.get(pointNum)+" should be "+correctLabel);
                numErrors++;
            }
            pointNum++;
        }
        br.close();
        double errorRate = (numInputPoints - numErrors)/((double)numInputPoints);
        System.out.println("Classified "+(numInputPoints-numErrors)+" correct out of "+numInputPoints+" ("+(errorRate*100)+"%)");
    }

    /**
     * Writes the resulting labels of classifying the result to outFileName in csv format
     * @param  outFileName           filename that will contain the output labels
     * @throws FileNotFoundException 
     * @throws IOException           
     */
    public void outputResult(String outFileName) throws FileNotFoundException, IOException{
        BufferedWriter bw = new BufferedWriter(new FileWriter(outFileName));
        System.out.println("Outputing classification to "+outFileName);

        for(int i=0; i<numInputPoints; i++){
            bw.write(String.valueOf(this.outputLabels.get(i)));
            bw.newLine();
        }

        bw.close();
        System.out.println("Wrote "+ numInputPoints+" points to file.");
    }

    /**
     * Picks k elements without replacement from the an array of length len randomly and returns an array of their indices
     * We only need the length since this will pick k unique indices at random from within that array
     * Basically a reservoir sample algorithm
     *
     * Precondition: k is greater than len
     * 
     * @param  k the number of elements to select
     * @param  len The length of the array to sample from
     * @return   Array of the indices of the selected elements (all unique)
     */
    private int[] sampleWithoutReplacement(int k, int len){
        int[] result = new int[k];

        // Fill the reservoir array
        int i = 0;
        for(; i<k; i++){
            result[i] = i;
        }

        Random r = new Random(System.currentTimeMillis());
        // sample from the remaining training points
        for(; i<len; i++){
            int j = r.nextInt(i);
            if(j < k){
                result[j] = i;
            }
        }

        return result;
    }

    /**
     * Picks k elements with replacement from an array of length len and returns an array of their indices
     * @param k   The number of elements to select 
     * @param len The length of the array to sample from
     * @return    An array of k indices (some may be duplicates)
     */
    private int[] sampleWithReplacement(int k, int len){
        int[] result = new int[k];

        Random r = new Random(System.currentTimeMillis());
        for(int i=0; i<k; i++){
            int j = r.nextInt(len);
            result[i] = j;
        }

        return result;
    }

    /**
     * Computes the entropy of the distribution
     * Precondition: distribution must be normalized
     * @param  distribution The probability distribution
     * @return              The entropy of the distribution
     */
    private double entropy(double[] distribution){
        double res = 0.0;
        for(int i=0; i<distribution.length; i++){
            if(distribution[i] == 0.0){
                res += 0.0;
            }else{
                res += (-1*distribution[i]*Math.log(distribution[i]));
            }
        }

        return res;
    }

    /**
     * Returns the information gain by splitting s on splitIdx;
     * Precondition: s is an array of the features and in increasing sorted order
     * @param  s        indices of the training points in set s
     * @param  boundary the condition to separate on, ie xi smaller than t
     * @param  fnum     the feature index we are considering
     * @return          the information gain of spliting on that index
     */
    private double informationGain(int[] s, double boundary, int fnum){
        // First compute the probability distributions
        double[] distLeft = new double[2];
        double[] distRight = new double[2];
        double[] distAll = new double[2];

        int leftSize = 0, rightSize = 0;

        for(int i=0; i<s.length; i++){
            double featureVal = this.trainingVectors.get(s[i])[fnum];
            int pointLabel = this.trainingLabels.get(s[i]);
            // sanity check
            assert pointLabel >= 0 && pointLabel < 2;
            if(featureVal <= boundary){
                distLeft[pointLabel] += 1;
                leftSize++;
            }
            else{
                distRight[pointLabel] += 1;
                rightSize++;
            }
            distAll[pointLabel] += 1;
        }

        // Normalize all the distributions
        double leftSum = (distLeft[0]+distLeft[1]);
        double rightSum = (distRight[0]+distRight[1]);
        double allSum = (distAll[0]+distAll[1]);
        for(int i=0; i<2; i++){
            distLeft[i] = distLeft[i] / leftSum;
            distRight[i] = distRight[i] / rightSum;
            distAll[i] = distAll[i] / allSum;
        }

        double result = entropy(distAll);
        double leftRatio = ((double)leftSize)/s.length;
        double rightRatio = ((double)rightSize)/s.length;
        result = result - (leftRatio*entropy(distLeft) + rightRatio*entropy(distRight));
        return result;
    }

    /**
     * A Decision Tree Node class
     */
    class DecisionTreeNode{

        DecisionTreeNode left;
        DecisionTreeNode right;
        int featureNum;         //The feature index this node splits on
        double boundary;        //The boundary of the feature value this node splits on
        int label;              //If this node is a leaf then this is the resulting label from arriving to this leaf
        boolean isLeaf;

        /**
         * Recursively constructs a Decision Tree rooted at this node from this set
         * Precondition: Set has length atleast one
         * @param  set                  the indices of the training points in trainingVectors
         * @param  parentMajorityLabel  In the event that the set passed to the constructor has no elements, then we create a leaf with the most common label from the parent set
         * @param  depth                the maximum depth after which we create a leaf and pick the majority label from the set of training points
         */
        public DecisionTreeNode(int[] set, int parentMajorityLabel, int depth){
            // Stopping condition: set has length 0, create a leaf with label equal to parent majority label
            if(set.length == 0){
                this.isLeaf = true;
                this.label = parentMajorityLabel;
                return;
            }

            // First create an array of all the values
            int[] setLabels = new int[set.length];
            int zeroCount = 0;
            boolean allSame = true; //wether all the labels are the same
            for(int i=0; i<set.length; i++){
                setLabels[i] = trainingLabels.get(set[i]);
                if(i != 0)          // Keeps track if all the levels are the same
                    allSame = allSame && (setLabels[i-1] == setLabels[i]);

                if(setLabels[i] == 0) //Count how many of the labels are zeroes
                    zeroCount++;
            }
            
            // Find the majority label in the set (we will pass this down in the recursive call to DecisionTreeNode)
            int majorityLabel = (zeroCount > (set.length/2)) ? 0 : 1;

            // Stopping condition: if all elements in set have the same label.
            // This will also stop a set with only 1 element
            if(allSame){
                this.label = setLabels[0];
                this.left = this.right = null;
                this.boundary = 0.0;
                this.featureNum = -1;
                this.isLeaf = true;
                return;             
            }

            // Stopping condition: we reached the maximum depth so we just make this a leaf node and assign the majority label
            if(depth == MAX_TREE_DEPTH){
                this.label = majorityLabel;
                this.right = this.left = null;
                this.isLeaf = true;
                return;
            }

            // FEATURE SAMPLING: Pick sqrt(numFeatures) as candidates to split on
            int numFeatures = (int) Math.sqrt((double) VECTOR_SIZE);
            int[] featureIndices = sampleWithoutReplacement(numFeatures, VECTOR_SIZE);

            // Keep track of the best feature to split on
            int bestSplitFeature = -1;
            double bestInformationGain = 0.0;
            double bestSplitBoundary = 0.0;

            // Try each feature as splitting point
            for(int i=0; i<numFeatures; i++){
                int featureIdx = featureIndices[i];
                // build the list of feature values for the elements in the set
                // remove duplicate feature vals
                ArrayList<Double> featureVals = new ArrayList<Double>(set.length);
                for(int j=0; j<set.length; j++){
                    // get feature number i from training point j
                    int trainingVectorIdx = set[j];
                    double val = trainingVectors.get(trainingVectorIdx)[featureIdx];
                    if(!featureVals.contains(val)){
                        featureVals.add(val);
                    }
                }

                // sort the feature values
                Collections.sort(featureVals);
                
                // Now try each possible boundary for splitting on this feature
                double boundary;
                // These will keep track of the best boundary point
                double best = 0.0;
                double bestBoundary = 0.0;
                for(int j=1; j<featureVals.size(); j++){
                    boundary = (featureVals.get(j-1)+featureVals.get(j))/2; //midpoint between the two vals
                    // Compute information gain
                    double ig = informationGain(set, boundary, featureIdx);

                    if(ig > best){
                        bestBoundary = boundary;
                        best = ig;
                    }
                }


                if(best > bestInformationGain){
                    bestInformationGain = best;
                    bestSplitFeature = featureIdx;
                    bestSplitBoundary = bestBoundary;
                }
            }

            // Stopping Condition: If no feature to split on then make a leaf node with the most common label in the set
            if(bestSplitFeature == -1){
                // Make a leaf with the most common label
                this.left = this.right = null;
                this.isLeaf = true;
                int count = 0;
                // find the most common label
                for(int i=0; i<setLabels.length; i++){
                    if(setLabels[i] == 0){
                        count++;
                    }
                }

                this.label = (count > (setLabels.length/2)) ? 0 : 1;
                return;              
            }

            // Split on the best feature, splitting boundary
            // Count how many elements will be in each set
            int leftSize = 0;
            for(int i=0; i<set.length; i++){
                int trainingPointIdx = set[i];
                double val = trainingVectors.get(trainingPointIdx)[bestSplitFeature];
                if(val <= bestSplitBoundary){
                    leftSize++;
                }
            }

            // Split the set into sLeft and sRight based on the best feature, best splitting boundary
            int[] sLeft = new int[leftSize];
            int[] sRight = new int[set.length - leftSize];
            int l=0,r=0; 
            for(int i=0; i<set.length; i++){
                int trainingPointIdx = set[i];
                double val = trainingVectors.get(trainingPointIdx)[bestSplitFeature];
                if(val <= bestSplitBoundary){
                    sLeft[l] = trainingPointIdx;
                    l++;
                }else{
                    sRight[r] = trainingPointIdx;
                    r++;
                }
            }

            // Set field appropriately
            this.boundary = bestSplitBoundary;
            this.isLeaf = false;
            this.featureNum = bestSplitFeature;
            this.left = new DecisionTreeNode(sLeft, majorityLabel, depth+1);
            this.right = new DecisionTreeNode(sRight, majorityLabel, depth+1);
            this.label = -1;
        }
    }


    public static void main(String[] args){
        if(args.length < 4){
            System.out.println("Usage: RandomForest num_trees bag_multiplier tree_depth input_vectors check_labels");
            System.exit(0);
        }

        String trainingVectorFilename = "emailDataset/trainFeatures.csv";
        String trainingLabelFilename = "emailDataset/trainLabels.csv";
        String inputVectorsFilename = args[3];
        boolean checkCorrectness = false;
        String validationLabelsFilename = "";
        if(args.length > 4){
            validationLabelsFilename = args[4];
            checkCorrectness = true;
        }
        int num_trees = Integer.parseInt(args[0]);
        double bag_mult = Double.parseDouble(args[1]);
        int max_trees = Integer.parseInt(args[2]);

        try{
            long start = System.currentTimeMillis();
            RandomForest rf = new RandomForest(trainingVectorFilename, trainingLabelFilename, num_trees, bag_mult, max_trees);
            System.out.println("Done reading files");
            rf.train();
            rf.readInputVectors(inputVectorsFilename);
            rf.classifyInput();

            if(checkCorrectness){
                rf.checkOutput(validationLabelsFilename, false);
            }

            rf.outputResult("RandomForestOut.csv");

            long end = System.currentTimeMillis();

            System.out.println("Total runtime: "+(end-start)/1000+" s");
            System.out.println("num trees: "+ num_trees);
            System.out.println("bag multiplier: "+bag_mult);
            System.out.println("max tree depth: "+max_trees);
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}