// Report Guidelines
// [ ] - Font Name: Times New Roman
// [x] - Font Size: 12 pt (for normal text)
// [x] - Left Margin: 1.25 inch
// [x] - Right Margin: 1 inch
// [x] - Top Margin: 1 inch
// [x] - Bottom Margin: 1 inch
// [x] - Header and Footer: 0.5 inch
// [~] - Line Spacing: 1.5
// [x] - All the text should be justified.
// [x] - Heading should be in following standard
// [x] - 1. Heading1 (16 pt, Bold)
// [x] - 1.1 Heading2 (14 pt, Bold)
// [x] - 1.1.1 Heading3 (13 pt, Bold)
// [x] - 1.1.1.1 Heading4 (12 pt, Bold)

// Numbering sections, subsections, equations, figures etc. –  
// •  A word on numbering scheme used in the project is in order. It is common practice to use decimal numbering in the project. If the chapter number is 2, the section numbers will be 2.1,2.2, 2.3 etc. The subsections in section 2.2 will be numbered as 2.2.1, 2.2.2 etc. Unless essential, it is not necessary to use numbers to lower levels than three stages.  
// •  Similarly, it is useful and convenient to number the figures also chapter-wise. The figures in chapter 4 will be numbered as Figure 4.1: Figure Name, Figure 4.2: Figure Name etc. This helps you in assembling the figures and putting it in proper order. Similarly, the tables are also numbered as Table 4.1: Table Name, Table 4.2: Table Name etc. All figures and tables should have proper captions. Usually, the figure captions are written below the figure and table captions on top of the table. All figures should have proper description by legends, title of the axes and any other information to make the figures self-explanatory.  
// •  The same numbering scheme can be used for equations also. Only thing to be remembered is that references to the figures are made like Figure 4.2: Figure Name and equations as Eqn (5.8). 


#import "@preview/gantty:0.2.0": gantt
#import "@preview/timeliney:0.2.1"

#set page(
  paper: "a4", 
  margin: (left: 1.25in, right: 1in, top: 1in, bottom: 1in),
  header-ascent: 0.5in,
  footer-descent: 0.5in,
)

#let submit_date = datetime(year: 2082, month: 02, day: 02).display()
#let line_spacing = 1.145em
#let first_heading_spacing = 1.5em

#set text(size: 12pt, font: "Times New Roman")
#set par(justify: true, leading: line_spacing)

#show heading.where(level: 1): it => [
  #set text(size: 16pt, weight: "bold")
  #set align(center)
  #block(spacing: first_heading_spacing, it)
]

#show heading.where(level: 2): it => block(spacing: line_spacing)[
  #set text(size: 14pt, weight: "bold")
  #it
]

#show heading.where(level: 3): it => block(spacing: first_heading_spacing)[
  #set text(size: 13pt, weight: "bold")
  #it
]
#show heading.where(level: 4): it => block(spacing: line_spacing)[
  #set text(size: 12pt, weight: "bold")
  #it
]

#set figure(numbering: num => 
  (counter(heading).get() + (num,)).map(str).join("."))

// Cover page
#align(center+horizon)[
  #set par(leading: 1.5em)
  #text(size: 14pt)[*A Project Proposal On*]\
  #text(size: 20pt)[*Early Parkinson's Detection from \ Voice Dysphonia*]
  #v(1.2cm)
  #image("img/pu_logo_older.png", width: 30%)
  #v(1.2cm)
  #text(size: 14pt)[
    Submitted in the Partial Fulfillment of the\
    Requirements for the Degree of Bachelor of Software Engineering\
    Awarded by Pokhara University
  ]
  #v(1.2cm)
  *Submitted By:*\
  *
    Diwas Rimal (22180069)\
    Nischal Bastola (22180076)\
    Sushant Baral (22180093)\
    Sworup Raj Paudel (22180095)\
  *
  #v(1.2cm)
  *
  #text(size: 20pt)[School of Engineering]\
  #text(size: 16pt)[Faculty of Science and Technology]\
  #text(size: 16pt)[POKHARA UNIVERSITY]\
  #text(size: 18pt)[May 2025]\
  *
]

#pagebreak()
#set page(numbering: "i")
#counter(page).update(1)

= ABSTRACT
We are building a machine learning model that helps early diagnosis of the Parkinson's disease. Parkinson's disease is a neurological disorder that affects the human body's motor capabilities. Although motor symptoms can help diagnose the disease in earlier stages, checks of such symptoms require presence of a health professional. It would be helpful if one could check themselves how likely they may be having early symptoms of Parkinson's. We are considering the fact that one of the early symptoms of the Parkinson's also includes degradation of person's voice, and using this fact to build a binary classification machine learning model that helps its early diagnosis by taking as input the voice of the person. This can benefit the medical industry as well as normal people, who can't afford regular visits to health professionals. We plan to train the model using existent data, then also create a user interface where users can input their voice and check the likelihood of them having early Parkinson's symptoms.

\
*Key Words*: _Parkinson's Disease, Machine Learning, Voice Analysis_

#pagebreak()
= TABLE OF CONTENTS

#outline(title: none)

#pagebreak()
= LIST OF FIGURES
#outline(title: none, target: figure.where(kind: image))

#pagebreak()
= LIST OF TABLES
#outline(title: none, target: figure.where(kind: table))

#pagebreak()
= ABBREVIATIONS
#figure(
  table(
    columns: (0.3fr, 1fr), // todo: both columns equal? (1fr, 1fr)
    // note: yo list alphabetical order ma sorted hunu parxa hai - diwas
    "CNNs", "Convolutional Neural Networks",
    "DFA", "Detrended Fluctuation Analysis",
    "HNR", "Harmonic to Noise Ratio",
    "IEEE", "Institute of Electrical and Electronics Engineers",
    "MFCCs", "Mel-Frequency Cepstral Coefficients",
    "ML", "Machine Learning",
    "PD", "Parkinson's Disease",
    "PPE", "Pitch Period Entropy",
    "RNNs", "Recurrent Neural Networks",
    "RPDE", "Recurrence Period Density Entropy",
    "SVM", "Support Vector Machine",
    "UCI", "University of California, Irvine",
  ),
  caption: "Table of abbreviations",
)

#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)

#set heading(numbering: (..n) => 
  if n.pos().len() > 1 {
    return numbering("1.", ..n)
  }
)
  
= CHAPTER 1
= INTRODUCTION
#counter(heading).update(1)
== Background
Parkinson’s Disease is a disorder that affects movement and speech. In the early stages, it can cause small changes in a person’s voice that are hard to notice. These voice changes can be used to detect the disease early. With the help of machine learning, we can analyze a person’s voice and find patterns that may indicate Parkinson’s.This project uses voice analysis and machine learning techniques to make early detection easier, faster, and more accessible.

==  Problem Statement
Parkinson’s Disease is often diagnosed at a later stage because early symptoms are difficult to detect.Traditional methods can be costly, time-consuming, and require a clinical setting. There is a need for a simple, low-cost, and non-invasive method to detect Parkinson’s at an early stage. This project aims to solve that problem by using voice analysis and machine learning to identify early signs of the disease.

==  Objectives
The objectives of this project are as follows:
- To build an intelligent system that takes voice input, extracts acoustic features (like jitter, shimmer, and HNR), and uses machine learning to predict the likelihood of Parkinson’s Disease.
- To enable people to check early stages of Parkinson's Disease in remote areas where access to medical facilities might be limited.

==  Applications
The applications of this project are:
- *Medical Screening:* Helps doctors with early detection of Parkinson’s Disease using voice tests.
- *Remote Health Monitoring:* Useful for patients in rural or remote areas where access to neurologists is limited.
- *Self-Assessment Tool:* Individuals can check for early signs of Parkinson’s at home using a computer or mobile device.

== Project Features
- *Voice Input:* Users can record or upload their voice for analysis.
- *Feature Extraction:* The system extracts acoustic features like jitter, shimmer, and HNR from the voice.
- *Prediction:* The system predicts the likelihood of Parkinson’s Disease based on voice patterns using machine learning.
- *Web Interface:* A user-friendly, web-based platform to make the process simple and accessible.

== Feasibility Analysis
=== Economic Feasibility
The project leverages open-source tools and datasets, making it cost-effective. It does not require specialized hardware, reducing costs. Can be deployed on cloud platforms with minimal costs, making it affordable for widespread use.

=== Technical Feasibility
The technologies required (Python, Scikit-learn, Parselmouth, Flask) are reliable and well-documented. Voice feature extraction and machine learning models (Random Forest, SVM, Logistic Regression) are feasible for this task with available resources. The project is scalable for deployment on different devices, including mobile and web platforms.

=== Operational Feasibility
The system is easy to use and accessible through a web interface, making it suitable for both medical professionals and general users. It can be implemented on a large scale, especially in health camps or remote monitoring scenarios. The system requires minimal training for users, ensuring a smooth adoption process.

== System Requirement

=== Software Requirement
- *Operating system:* Windows / macOS / Linux
- *Programming language:* Python 3.10+
- *Python libraries:*
  - numpy, pandas – for data handling
  - scikit-learn – for machine learning
  - parselmouth – for speech feature extraction
  - librosa – for advanced audio processing
  - matplotlib, seaborn – for visualization
  - streamlit or Flask – for building the web interface
- *Web Browser:* Chrome, Firefox, or any modern browser

=== Hardware Requirement
- *Processor:* Minimum dual-core CPU (Intel i3 or equivalent)
- *RAM:* Minimum 4 GB (8 GB recommended for smoother performance)
- *Storage:* At least 500 MB of free disk space
- *Microphone:* Decent-quality microphone (minimum 44.1 kHz sampling rate recommended) for recording clear voice samples

#pagebreak()
= CHAPTER 2
= LITERATURE REVIEW
#counter(heading).update(2)

== Voice as a Biomarker for Parkinson's Disease
Parkinson's Disease (PD) affects approximately 90% of patients through voice and speech disorders, collectively termed dysphonia. These manifestations often appear in early disease stages, sometimes predating traditional motor symptoms by several years. Little et al. @little2009suitability were among the first to establish that vocal impairments in PD patients are detectable through acoustic analysis, noting that phonation is particularly affected due to the reduced control of laryngeal muscles. Tsanas et al. @tsanas2010accurate further demonstrated that voice deterioration correlates with disease progression, making vocal biomarkers valuable for both early detection and longitudinal monitoring.

The primary vocal characteristics affected by PD include reduced loudness (hypophonia), breathiness, roughness, monotonicity (reduced pitch variation), and imprecise articulation. These symptoms result from the pathophysiological changes in PD, including rigidity, bradykinesia, and tremor affecting the speech apparatus. Orozco-Arroyave et al. @orozco2018neurospeech emphasized that these vocal impairments manifest consistently across languages and cultural backgrounds, confirming voice analysis as a robust, non-invasive, and cost-effective biomarker for PD screening.

== Acoustic Features and Signal Processing Methods
The extraction and analysis of acoustic features from voice recordings form the foundation of PD detection systems. Research has identified several categories of voice parameters that effectively differentiate between PD and healthy subjects:

Traditional acoustic measures include jitter (cycle-to-cycle variations in fundamental frequency), shimmer (amplitude variations), and harmonics-to-noise ratio (HNR). Sakar et al. @sakar2013collection demonstrated that these time-domain features alone can achieve moderate classification accuracy but are most effective when combined with other feature types.

Nonlinear dynamics measures, introduced by Little et al. @little2009suitability, have proven particularly valuable. These include recurrence period density entropy (RPDE), detrended fluctuation analysis (DFA), and correlation dimension, which capture the subtle nonlinear patterns in vocal fold vibrations disrupted by PD. Tsanas et al. @tsanas2011nonlinear expanded this approach with additional nonlinear measures, showing they outperform traditional acoustic parameters.

Spectral and cepstral features, including Mel-frequency cepstral coefficients (MFCCs), formant frequencies, and spectral flux, have been extensively studied by Vaiciukynas et al. @vaiciukynas2017detecting. These features capture the changes in vocal tract configuration and resonance characteristics affected by PD.

== Machine Learning Approaches for PD Classification
The evolution of machine learning techniques has significantly enhanced PD detection accuracy from voice recordings. Early studies predominantly employed conventional classifiers such as Support Vector Machines (SVM), Random Forests, and k-Nearest Neighbors. Little et al. @little2009suitability achieved 91.4% classification accuracy using SVM with a kernel-based approach on their dysphonia feature set.

Ensemble methods have shown promising results, with Sakar et al. @sakar2013collection demonstrating that combining multiple classifiers improves robustness against variability in voice recordings. Their approach using bootstrap aggregating (bagging) with decision trees achieved accuracy rates exceeding 92%.

More recently, deep learning approaches have gained prominence. Hemmerling et al. @hemmerling2016automatic implemented Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) directly on spectrograms and raw audio signals, eliminating the need for handcrafted feature extraction. Their deep neural network architecture achieved up to 95% accuracy on multiple datasets. Tracy et al. @tracy2020smartphone further refined these approaches for smartphone-based applications, enabling real-time PD screening with minimal computational resources.

#pagebreak()
= CHAPTER 3 
= METHODOLOGY
#counter(heading).update(3)

== Data Collecting
The data used in this research is taken from the UCI Parkinson's Dataset @parkinsons_uci_dataset. The original study presented feature extraction methods for general voice disorders The study included voice recordings from 31 people, including 23 people with Parkinson’s
Disease (PD) (16 males and 7 females) and eight Healthy Controls (males = 3 and females = 5). The dataset contains 195 records, 24 columns including a series of
biomedical voice measurements.

== Data Preprocessing
Before training the machine learning model, the collected dataset undergoes several preprocessing steps to ensure data quality and model performance:
  - *Handling Missing Values:* Checked for any missing or null values in the dataset and applied imputation techniques if needed.
  - *Normalization/Scaling:* Since the dataset contains features with different units and ranges, normalization (e.g., Min-Max Scaling or Standardization) is applied to bring all features to a similar scale, improving model performance.
  - *Label Encoding:* If categorical values like gender are used, they are converted into numeric format using encoding techniques.
  - *Train-Test Split:* The dataset is divided into training and testing sets (e.g., 80% training, 20% testing) to evaluate model performance on unseen data.
  
== Feature Extraction
For real-time user input, Parselmouth is used to extract acoustic features like Jitter, Shimmer, HNR, RPDE, DFA, and PPE. The UCI Parkinson's dataset @parkinsons_uci_dataset already contains these pre-extracted features. These features help in detecting vocal instability associated with Parkinson’s.

== Model Building

First we will train models using ML algorithms like K-nearest neighbors,  Support Vector Machine (SVM),  Random Forest,  Decision tree,  Logistic Regression. We then can use cross-validation techniques to evaluate the model accuracy and then select the best-performing model based on metrics like accuracy, precision, recall and F1-score. Python Libraries scikit-learn, numpy, matplotlib, pandas are used for model building and visualization.

// - Train ML models such as K-nearest neighbors,  Support Vector Machine (SVM),  Random Forest,  Decision tree,  Logistic Regression.
// - Use cross-validation to evaluate model accuracy
// - Select the best-performing model based on metrics (accuracy, precision, recall).
// - Python Libraries scikit-learn, numpy, matplotlib, pandas are used for model building and visualization.


==  System Integration

The built ML model is utilized via an web interface, taking inputs from and web app built using Streamlit or Flask, which allows users to record/upload voice, then extract features and predict results in real-time. The integration process involves integrating built ML model with the web app.

// - Build a web app using Streamlit or Flask.
// - Allow users to record/upload voice, then extract features and predict results in real-time.

== Output
The system displays the likelihood of Parkinson’s (e.g., “Low Risk”, “High Risk”) along with confidence scores. Optionally, display feature values for transparency.

#pagebreak()
= CHAPTER 4 
= SYSTEM ANALYSIS AND DESIGN
#counter(heading).update(4)
== System Analysis

=== System Objectives
- To develop a web-based system that predicts the likelihood of Parkinson’s Disease using voice input.
- To extract acoustic features such as jitter, shimmer, and HNR and feed them into a machine learning model for accurate diagnosis.

=== System Scopes
// - Accepts user-recorded or uploaded voice samples.
// - Automatically extracts relevant features using
// - Processes data through a trained machine learning model to provide a prediction.
// - Displays prediction results through an easy-to-use web interface.
// - Useful for early screening, especially in remote areas.
The system should accept user-recorded or uploaded voice samples, making it accessible and flexible for different users. It automatically extracts relevant features from the audio and processes the data through a trained machine learning model to generate predictions. These prediction results are then displayed through a user-friendly web interface. This setup is particularly useful for early screening purposes, especially in remote or underserved areas where access to medical facilities may be limited.

=== Key Functionalities
- Voice record/upload interface.
- Feature extraction engine (jitter, shimmer, HNR, etc.).
- Parkinson’s prediction using a trained ML model.
- Result display with simple interpretation.

==  Requirement Analysis

=== Functional Requirements
- The system must allow users to record/upload their voice.
- It must extract voice features using Python libraries like parselmouth.
- It must classify the input as likely Parkinson’s or not using an ML model.
- It should show results with user-friendly feedback.

=== Non Functional Requirements
// - *Performance:* The system should return results within a few seconds.
// - *Usability:* Simple interface suitable for non-technical users.
// - *Reliability:* Consistent results for similar inputs.
// - *Scalability:* Should handle multiple users.//online host hanni ho vne
// - *Portability:* Can run on different operating systems and devices (desktop/mobile).
The system should deliver high performance by returning results within a few seconds. It must prioritize usability with a simple and intuitive interface that caters even to non-technical users. Reliability is essential, ensuring consistent results for similar inputs across sessions. To support growth, the system should be scalable, capable of handling multiple users simultaneously, especially if hosted online. Additionally, portability is important, allowing the system to run smoothly across various operating systems and devices, including both desktops and mobile platforms.

== System Design
===   System Design/Architecture

#figure(
  image("img/system-design.png"),
  caption: [System Architecture Diagram]
)

===   Use Case Diagram
#figure(
  image("img/use-case-diagram.png"),
  caption: [Use Case Diagram]
)


===   Process Modeling Diagram
#figure(
  image("img/process-modeling-diagram.png",  width: 180%),
  caption: [Process Modeling Diagram]
)

// ===   Sequence Diagram
// #figure(
//   image("img/sequence-diag.svg", width: 110%),
//   caption: [Sequence Diagram]
// )


#pagebreak()
= CHAPTER 5 
= EPILOGUE
#counter(heading).update(5)
== Expected Output
An web application that can take audio sample of a person and determine whether the person has early symptoms of Parkinson's using a trained machine learning model having reasonable accuracy.

== Budget Analysis
The development of this project will have no costs, since every tool used will be cost-free. Running the product however, will not be cost-free as it will need backend server and domain.

#align(center)[
  #figure(
    table(
      columns: (auto, auto),
      align: center,
      table.header(
        [*Component*], [*Cost (Rs.)*],
      ),
      "Domain (.com)", "2000/year",
      "Server", "3500/year",
    ),
    caption: [Product running costs]
  )
]

== Work Schedule
#figure(
  // gantt(yaml("gantt.yaml")),
  image("img/gantt.png", width: 100%),
  caption: [Gantt Chart],
)

#pagebreak()
= CHAPTER 6
= REFERENCES
#counter(heading).update(6)
#bibliography("references.bib", style: "ieee", title: none)

// #pagebreak()
// = CHAPTER 7
// = APPENDICES
// #counter(heading).update(7)
