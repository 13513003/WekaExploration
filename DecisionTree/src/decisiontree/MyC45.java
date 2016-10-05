/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;
import weka.core.AttributeStats;

/**
 *
 * @author Asus
 */
public class MyC45 extends Classifier {
    /** for serialization */
    static final long serialVersionUID = -2693678647096322561L;

    /** The node's successors. */ 
    private MyC45[] m_Successors;

    /** Attribute used for splitting. */
    private Attribute m_Attribute;

    /** Class value if node is leaf. */
    private double m_ClassValue;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution;

    /** Class attribute of dataset. */
    private Attribute m_ClassAttribute;

    /**
     * Returns a string describing classifier
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {

      return  "Class for generating a pruned or unpruned C4.5 decision tree. For more "
        + "information, see\n\n"
        + "Ross Quinlan (1993). \"C4.5: Programs for Machine Learning\", "
        + "Morgan Kaufmann Publishers, San Mateo, CA.\n\n";
    }
    
    /**
    * Returns default capabilities of the classifier.
    *
    * @return      the capabilities of this classifier
    */
    public Capabilities getCapabilities() {
     Capabilities result = super.getCapabilities();
     result.disableAll();

     // attributes
     result.enable(Capability.NOMINAL_ATTRIBUTES);
     result.enable(Capability.NUMERIC_ATTRIBUTES);
     result.enable(Capability.MISSING_VALUES);

     // class
     result.enable(Capability.NOMINAL_CLASS);
     result.enable(Capability.MISSING_CLASS_VALUES);

     // instances
     result.setMinimumNumberInstances(0);

     return result;
    }
    
    // TODO: PruneTree
    // TODO: Calculate Gain Ratio
    // TODO: Handle Numeric and Nominal Values
    public void buildClassifier (Instances instances) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // handle instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        
        // handle missing values
        Instances processedInstances = handleMissingValues(instances);

        makeTree(processedInstances);
    }
    
    /**
    * Method for building an C45 tree.
    *
    * @param instances the training data
    * @exception Exception if decision tree can't be built successfully
    */
    private void makeTree(Instances instances) throws Exception {

     // Check if no instances have reached this node.
     if (instances.numInstances() == 0) {
       m_Attribute = null;
       m_ClassValue = Instance.missingValue();
       m_Distribution = new double[instances.numClasses()];
       return;
     }

     // Compute attribute with maximum gain ratio.
     double[] gainRatios = new double[instances.numAttributes()];
     Enumeration attrEnum = instances.enumerateAttributes();
     while (attrEnum.hasMoreElements()) {
       Attribute attr = (Attribute) attrEnum.nextElement();
       // TODO: 2 types of computeGainRatio to handle numeric and nominal values
       gainRatios[attr.index()] = computeGainRatio(instances, attr);
     }
     m_Attribute = instances.attribute(Utils.maxIndex(gainRatios));

     // Make leaf if gain ratio is zero. 
     // Otherwise create successors.
     if (Utils.eq(gainRatios[m_Attribute.index()], 0)) {
       m_Attribute = null;
       m_Distribution = new double[instances.numClasses()];
       Enumeration instEnum = instances.enumerateInstances();
       while (instEnum.hasMoreElements()) {
         Instance inst = (Instance) instEnum.nextElement();
         m_Distribution[(int) inst.classValue()]++;
       }
       Utils.normalize(m_Distribution);
       m_ClassValue = Utils.maxIndex(m_Distribution);
       m_ClassAttribute = instances.classAttribute();
     } else {
       Instances[] splitData = splitData(instances, m_Attribute);
       m_Successors = new MyC45[m_Attribute.numValues()];
       for (int j = 0; j < m_Attribute.numValues(); j++) {
         m_Successors[j] = new MyC45();
         m_Successors[j].makeTree(splitData[j]);
       }
     }
    }
    
    /**
    * Splits a dataset according to the values of a nominal attribute.
    *
    * @param data the data which is to be split
    * @param att the attribute to be used for splitting
    * @return the sets of instances produced by the split
    */
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
     for (int i = 0; i < splitData.length; i++) {
       splitData[i].compactify();
     }
     return splitData;
    }
    
    /**
    * Computes information gain for an attribute.
    *
    * @param data the data for which info gain is to be computed
    * @param att the attribute
    * @return the information gain for the given attribute and data
    * @throws Exception if computation fails
    */
    private double computeGainRatio(Instances instances, Attribute attr) throws Exception {
        double gainRatio = computeEntropy(instances);
        Instances[] splitData = splitData(instances, attr);
        for (int j = 0; j < attr.numValues(); j++) {
          if (splitData[j].numInstances() > 0) {
            gainRatio -= ((double) splitData[j].numInstances() /
                         (double) instances.numInstances()) *
              computeEntropy(splitData[j]);
          }
        }
        return gainRatio;
    }
    
    /**
    * Computes the entropy of a dataset.
    * 
    * @param data the data for which entropy is to be computed
    * @return the entropy of the data's class distribution
    * @throws Exception if computation fails
    */
    private double computeEntropy(Instances data) throws Exception {

     double [] classCounts = new double[data.numClasses()];
     Enumeration instEnum = data.enumerateInstances();
     while (instEnum.hasMoreElements()) {
       Instance inst = (Instance) instEnum.nextElement();
       classCounts[(int) inst.classValue()]++;
     }
     double entropy = 0;
     for (int j = 0; j < data.numClasses(); j++) {
       if (classCounts[j] > 0) {
         entropy -= classCounts[j] * Utils.log2(classCounts[j]);
       }
     }
     entropy /= (double) data.numInstances();
     return entropy + Utils.log2(data.numInstances());
    }
    
    private Instances handleMissingValues(Instances data) {
        Instances newData = data;
        Enumeration attrEnum = newData.enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            if (attr.isNominal()) {
                int maxIdx = 0;
                AttributeStats attrStats = newData.attributeStats(attr.index());
                for (int i=0; i<attr.numValues(); i++) {
                    if (attrStats.nominalCounts[i] > attrStats.nominalCounts[maxIdx]) {
                        maxIdx = i;
                    }
                }

                for (int i=0; i<newData.numInstances(); i++) {
                    if (newData.instance(i).isMissing(attr.index())) {
                        newData.instance(i).setValue(attr.index(), maxIdx);
                    }
                }
            }
            else if (attr.isNumeric()) {
                double mean = newData.attributeStats(attr.index()).numericStats.mean;
                for (int i=0; i<newData.numInstances(); i++) {
                    if (newData.instance(i).isMissing(attr.index())) {
                        newData.instance(i).setValue(attr.index(), mean);
                    }
                }
            }
        }

        return newData;
    }
}
