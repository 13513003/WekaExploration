/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class myID3 extends Classifier{
    
    private myID3[] child;
    private Attribute splitAttr;
    private double leafValue;
    private double[] leafDist;
    private Attribute classAttr;
    private static final long serialVersionUID = 2404406538125671745L;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        data = new Instances(data);  
        data.deleteWithMissingClass();
        makeTree(data);
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities cap = super.getCapabilities();
        cap.disableAll();

        // attributes
        cap.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        cap.enable(Capability.BINARY_CLASS);
        cap.enable(Capability.NOMINAL_CLASS);
        cap.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        cap.setMinimumNumberInstances(0);

        return cap;
    }
    
    private void makeTree(Instances data) {
        // Check if no instances have reached this node.  
        if (data.numInstances() == 0) {  
          splitAttr = null;  
          leafValue = Instance.missingValue();  
          leafDist = new double[data.numClasses()];  
          return;  
        }
        
        if (data.numDistinctValues(data.classIndex()) == 1){
            leafValue = data.firstInstance().classValue();
            return;
        }

        // Compute attribute with maximum information gain.  
        double[] infoGains = new double[data.numAttributes()];  
        Enumeration attEnum = data.enumerateAttributes();  
        while (attEnum.hasMoreElements()) {  
          Attribute att = (Attribute) attEnum.nextElement();  
          infoGains[att.index()] = computeInfoGain(data, att);  
        }  
        splitAttr = data.attribute(Utils.maxIndex(infoGains));  

        // Make leaf if information gain is zero.   
        // Otherwise create successors.  
        if (Utils.eq(infoGains[splitAttr.index()], 0)) {  
          splitAttr = null;  
          leafDist = new double[data.numClasses()];  
          Enumeration instEnum = data.enumerateInstances();  
          while (instEnum.hasMoreElements()) {  
            Instance inst = (Instance) instEnum.nextElement();  
            leafDist[(int) inst.classValue()]++;  
          }  
          Utils.normalize(leafDist);  
          leafValue = Utils.maxIndex(leafDist);  
          classAttr = data.classAttribute();  
        } else {  
          Instances[] splitData = splitData(data, splitAttr);  
          child = new myID3[splitAttr.numValues()];  
          for (int j = 0; j < splitAttr.numValues(); j++) {  
            child[j] = new myID3();  
            child[j].makeTree(splitData[j]);  
          }  
        }  
    }
    
    private double computeInfoGain(Instances data, Attribute att) {   
        double infoGain = computeEntropy(data);  
        Instances[] splitData = splitData(data, att);  
        for (Instances split : splitData){
            if (split.numInstances() > 0){
                infoGain -= ((double )split.numInstances() / (double) data.numInstances())
                        * computeEntropy(split);
            }
        }
        return infoGain;  
    }

    private double computeEntropy(Instances data) {   
        int numClasses = data.numClasses();
        int[] classCount = new int[numClasses];
        ArrayList<Double> classValues = new ArrayList<>();
        Enumeration<Instance> instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance instance = instEnum.nextElement();
            double classValue = instance.classValue();
            if (!classValues.contains(classValue)) {
                classValues.add(classValue);
            }
            int index = classValues.indexOf(classValue);
            classCount[index]++;
        }
        double entropy = 0.0;
        for (Double value: classValues) {
            int index = classValues.indexOf(value);
            if (classCount[index] > 0) {
                double temp = (double) classCount[index] / (double) numClasses;
                entropy += (-1) * temp * Utils.log2(temp);
            }
        }
        return entropy;
    }
    
    private Instances[] splitData(Instances data, Attribute att) {  
        Instances[] splitData = new Instances[att.numValues()];  
        for (int j = 0; j < att.numValues(); j++) {  
          splitData[j] = new Instances(data, data.numInstances());  
        }
       
        Enumeration instEnum = data.enumerateInstances();  
        while (instEnum.hasMoreElements()) {  
          Instance inst = (Instance) instEnum.nextElement();  
          splitData[(int) inst.value(att)].add(inst);
        }
        for (Instances split : splitData) {
            split.compactify();
        }
        return splitData;  
    }
    
    public String toString() {  
        if ((leafDist == null) && (child == null)) {  
          return "Id3: No model built yet.";  
        }  
        return "Id3\n\n" + toString(0);  
    }
    
    private String toString(int level) {  
  
    StringBuffer text = new StringBuffer();  
      
    if (splitAttr == null) {  
      if (Instance.isMissingValue(leafValue)) {  
        text.append(": null");  
      } else {  
        text.append(": "+classAttr.value((int) leafValue));  
      }   
    } else {  
      for (int j = 0; j < splitAttr.numValues(); j++) {  
        text.append("\n");  
        for (int i = 0; i < level; i++) {  
          text.append("|  ");  
    }  
        text.append(splitAttr.name() + " = " + splitAttr.value(j));  
        text.append(child[j].toString(level + 1));  
      }  
    }  
    return text.toString();  
  } 
    
}
