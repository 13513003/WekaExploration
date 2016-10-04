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
    
    // TODO: Handle Missing Values
    // TODO: PruneTree
    // TODO: Calculate Gain Ratio
    // TODO: Handle Numeric and Nominal Values
    public void buildClassifier (Instances instances) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        makeTree(instances);
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

     // Compute attribute with maximum information gain.
     double[] infoGains = new double[instances.numAttributes()];
     Enumeration attEnum = instances.enumerateAttributes();
     while (attEnum.hasMoreElements()) {
       Attribute att = (Attribute) attEnum.nextElement();
       infoGains[att.index()] = computeInfoGain(instances, att);
     }
     m_Attribute = instances.attribute(Utils.maxIndex(infoGains));

     // Make leaf if information gain is zero. 
     // Otherwise create successors.
     if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
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
}
