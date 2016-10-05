package decisiontree;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;
import weka.core.converters.*;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.unsupervised.attribute.Remove;

public class Main {
    private static Instances dataSet;
    private static Classifier cls;
    private static String classifierType;
    
    public static void loadFile() throws Exception {
        ArrayList<String> fileNames = getFileNames("input/train");
        System.out.println("Available Training Set : ");
        for (int i = 0; i < fileNames.size(); i++) {
            System.out.println(i + ". " + fileNames.get(i));
        }
        System.out.print("Select the number of the training set to load : ");
        Scanner in = new Scanner(System.in);
        int choice = in.nextInt();
        if (choice > fileNames.size()-1) {
            System.out.println("Input error. Try again.");
            System.out.print("Select the number of the training set to load : ");
            choice = in.nextInt();
        }
        dataSet = ConverterUtils.DataSource.read("input/train/" + fileNames.get(choice));
         if (dataSet.classIndex() == -1)
            dataSet.setClassIndex(dataSet.numAttributes() - 1);
        System.out.println("Dataset "+ fileNames.get(choice) + "successfully loaded.\n");
    }
    
    public static ArrayList<String> getFileNames(String dir) {
        ArrayList<String> results = new ArrayList<>();
        File[] files = new File(dir).listFiles();

        for (File file : files) {
            if (file.isFile()) {
                results.add(file.getName());
            }
        }
        return results;
    }
    
    public static void command() throws Exception {
        if (classifierType != null) {
            System.out.println("Current Classifier : " + classifierType);
        }
        System.out.println("Commands :");
        System.out.println("1. Load New DataSet");
        System.out.println("2. Print DataSet");
        System.out.println("3. Remove Attribute");
        System.out.println("4. Apply Resample Filter");
        System.out.println("5. Create Classifier");
        if (classifierType != null) {
            System.out.println("6. Print Result");
            System.out.println("7. 10-Fold Cross Validation");
            System.out.println("8. Percentage Split");
            System.out.println("9. Use Training Set");
            System.out.println("10. Use Existing Test Set");
            System.out.println("11. Predict Test Set");
            System.out.println("12. Save Model");
            System.out.println("13. Load Model");
        }
        System.out.println("0. Exit");
        System.out.print("> ");
        Scanner in = new Scanner(System.in);
        int num = in.nextInt();
        if (classifierType == null) {
            if (num > 5) {
                System.out.println("Input error. Try again.");
                System.out.print("> ");
                num = in.nextInt();
            }
        } else {
            if (num > 13) {
                System.out.println("Input error. Try again.");
                System.out.print("> ");
                num = in.nextInt();
            }
        }
        switch(num) {
            case 1 : {
                loadFile();
                if (classifierType.equals("NB")) {
                    cls = new NaiveBayes();
                    cls.buildClassifier(dataSet);
                } else if (classifierType.equals("ID3")) {
                    cls = new myID3();
                    cls.buildClassifier(dataSet);
                } else if (classifierType.equals("C45")) {
                    cls = new MyC45();
                    cls.buildClassifier(dataSet);
                }
                break;
            }
            case 2 : {
                System.out.println(dataSet.toString());
                break;
            }
            case 3 : {
                System.out.print("Enter which attribute to remove : ");
                int index = in.nextInt();
                Attribute attribute = dataSet.attribute(index);
                dataSet.deleteAttributeAt(index);
                System.out.println("Removed attribute : " + attribute.toString());
                break;
            }
            case 4 : {
                System.out.println("Options : ");
                ArrayList<String> options = new ArrayList<>();
                in = new Scanner(System.in);
                System.out.print("-S (default 1) : ");
                String value = in.nextLine();
                if (value.isEmpty()) value = "1";
                options.add("-S");
                options.add(value);
                
                System.out.print("-Z (default 100) : ");
                value = in.nextLine();
                if (value.isEmpty()) value = "100";
                options.add("-Z");
                options.add(value);
                
                System.out.print("-B (default 0) : ");
                value = in.nextLine();
                if (value.isEmpty()) value = "0";
                options.add("-B");
                options.add(value);
                
                System.out.print("-no-replacement (default false) : ");
                value = in.nextLine();
                if (value.isEmpty()) value = "false";
                options.add("-no-replacement");
                options.add(value);
                
                if (options.get(7).equals("true")) {
                    System.out.print("-V (default false) : ");
                    value = in.nextLine();
                    if (value.isEmpty()) value = "false";
                    options.add("-V");
                    options.add(value);
                }
                
                String[] optionsArr = options.toArray(new String[options.size()]);
                Remove remove = new Remove();
                remove.setOptions(optionsArr);
                remove.setInputFormat(dataSet);
                dataSet = Resample.useFilter(dataSet, remove); 
                break;
            }
            case 5 : {
                System.out.println("Commands :");
                System.out.println("1. Naive-Bayes");
                System.out.println("2. Decision Tree - ID3");
                System.out.println("3. Decision Tree - C45");
                System.out.println("0. Return");
                System.out.print("> ");
                int choice = in.nextInt();
                switch(choice) {
                    case 1 : {
                        cls = new NaiveBayes();
                        cls.buildClassifier(dataSet);
                        classifierType = "NB";
                        System.out.println("Naive-Bayes Classifier successfully built.\n");
                        break;
                    }
                    case 2 : {
                        cls = new myID3();
                        cls.buildClassifier(dataSet);
                        classifierType = "ID3";
                        System.out.println("ID3 Classifier successfully built.\n");
                        break;
                    }
                    case 3 : {
                        cls = new MyC45();
                        cls.buildClassifier(dataSet);
                        classifierType = "C45";
                        System.out.println("C45 Classifier successfully built.\n");
                        break;
                    }
                    default : {
                        break;
                    }
                }
                break;
            }
            case 6 : {
                System.out.println(cls.toString());
                break;
            }
            case 7 : {
                Evaluation eval = new Evaluation(dataSet);
                eval.crossValidateModel(cls, dataSet, 10, new Random(1));   
                System.out.println(eval.toSummaryString("\nResults\n===========\n", false));
                System.out.println(eval.toClassDetailsString("\nResults\n===========\n"));
                System.out.println(eval.toMatrixString("\nResults\n===========\n"));
                break;
            }
            case 8 : {
                System.out.print("Percentage of Training Set : ");
                float percentTrain = in.nextFloat();
                if ((percentTrain <= 0.0) || (percentTrain >= 100.0)) {
                    System.out.println("Input error. Try again.");
                    System.out.print("> ");
                    percentTrain = in.nextFloat();
                }
                float percentTest = 100 - percentTrain;                        
                System.out.println("Percentage of Test Set : " + percentTest);
                int trainSize = (int) Math.round(dataSet.numInstances() * percentTrain / 100);
                int testSize = dataSet.numInstances() - trainSize;
                dataSet.randomize(new java.util.Random(0));
                Instances train = new Instances(dataSet, 0, trainSize);
                Instances test = new Instances(dataSet, trainSize, testSize);
                cls.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nResults\n===========\n", false));
                System.out.println(eval.toClassDetailsString("\nResults\n===========\n"));
                System.out.println(eval.toMatrixString("\nResults\n===========\n"));
                break;
            }
            case 9 : {
                Evaluation eval = new Evaluation(dataSet);
                eval.evaluateModel(cls, dataSet);
                System.out.println(eval.toSummaryString("\nResults\n===========\n", false));
                System.out.println(eval.toClassDetailsString("\nResults\n===========\n"));
                System.out.println(eval.toMatrixString("\nResults\n===========\n"));
                break;
            }
            case 10 : {
                ArrayList<String> fileNames = getFileNames("input/test");
                System.out.println("Available Tests : ");
                for (int i = 0; i < fileNames.size(); i++) {
                    System.out.println(i + ". " + fileNames.get(i));
                }
                System.out.print("Select the number of the test to load : ");
                int choice = in.nextInt();
                if (choice > fileNames.size()-1) {
                    System.out.println("Input error. Try again.");
                    System.out.print("Select the number of the test to load : ");
                    choice = in.nextInt();
                }
                Instances test = ConverterUtils.DataSource.read("input/test/" + fileNames.get(choice));
                if (test.classIndex() == -1)
                    test.setClassIndex(test.numAttributes() - 1);
                Evaluation eval = new Evaluation(dataSet);
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nResults\n===========\n", false));
                System.out.println(eval.toClassDetailsString("\nResults\n===========\n"));
                System.out.println(eval.toMatrixString("\nResults\n===========\n"));
                break;
            }
            case 11 : {
                ArrayList<String> fileNames = getFileNames("input/test");
                System.out.println("Available Tests : ");
                for (int i = 0; i < fileNames.size(); i++) {
                    System.out.println(i + ". " + fileNames.get(i));
                }
                System.out.print("Select the number of the test to load : ");
                int choice = in.nextInt();
                if (choice > fileNames.size()-1) {
                    System.out.println("Input error. Try again.");
                    System.out.print("Select the number of the test to load : ");
                    choice = in.nextInt();
                }
                Instances test = ConverterUtils.DataSource.read("input/test/" + fileNames.get(choice));
                if (test.classIndex() == -1)
                    test.setClassIndex(test.numAttributes() - 1);
                for (int i = 0; i < test.numInstances(); i++) {
                    double label = cls.classifyInstance(test.instance(i));
                    test.instance(i).setClassValue(label);
                    for (int j = 0; j < test.instance(i).numValues()-1; j++) {
                        System.out.print(test.instance(i).stringValue(j)+",");
                    }
                    System.out.println(test.instance(i).stringValue(test.instance(i).numValues()-1));
		}
                break;
            }
            case 12 : {
                System.out.print("Name of your model : ");
                Scanner nameScan = new Scanner(System.in);
                String name = nameScan.nextLine();
                File dir = new File("saves/" + classifierType + "/");
                dir.mkdirs();
                SerializationHelper.write("saves/" + classifierType + "/" + name + ".model", cls);
                System.out.println("File " + name + ".model successfully created.\n");
                break;
            }
            case 13 : {
                ArrayList<String> fileNames = getFileNames("saves/" + classifierType);
                System.out.println("Saved Models : ");
                for (int i = 0; i < fileNames.size(); i++) {
                    System.out.println(i + ". " + fileNames.get(i));
                }
                System.out.print("Select the number of the model to load : ");
                int choice = in.nextInt();
                if (choice > fileNames.size()-1) {
                    System.out.println("Input error. Try again.");
                    System.out.print("Select the number of the model to load : ");
                    choice = in.nextInt();
                }
                if (classifierType.equals("NB"))
                    cls = (NaiveBayes) SerializationHelper.read(
                            new FileInputStream("saves/NB/" + fileNames.get(choice)));
                else if (classifierType.equals("ID3"))
                    cls = (myID3) SerializationHelper.read(
                            new FileInputStream("saves/ID3/" + fileNames.get(choice)));
                else if (classifierType.equals("C45"))
                    cls = (MyC45) SerializationHelper.read(
                            new FileInputStream("saves/C45/" + fileNames.get(choice)));
                System.out.println(fileNames.get(choice) + " successfully loaded.\n");
                break;
            }
            case 0 : {
                System.exit(1);
                break;
            }
        }
    }

    public static void main(String[] args){
        try {
            loadFile();
            while (true) {
              command();  
            }
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
     
    }
}

