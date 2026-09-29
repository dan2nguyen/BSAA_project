# BSAA_project
DREAM TEAM

Project dscription
<img width="1129" height="676" alt="image" src="https://github.com/user-attachments/assets/e342d970-db9f-4e0b-9e9b-57fd63340bd9" />


Assignment 1 (DEADLINE: 9.10. 23:59)
▪ Read biosignal(s) from EDF files. Explore the data with filters, time-frequency transforms, visualizations and collect interesting findings/observations regarding the characteristics of the signal and signal quality.
▪ Split the data into a training and a test set (e.g. 80/20). Move the test data somewhere separate and don’t touch it until we get to the algorithm validation!
▪ Try to calculate ground-truth heart rate from ECG using the Pan-Tomkins beat detection algorithm (e.g. “sleepecg” Python package).
▪ Research scientific literature for existing algorithms and compare your observations with the published information.
▪ Make a choice for developing an algorithm: Either try to implement an already published algorithm or come up with your own.
Deliverable (up to 10 points)
▪ Prepare a short presentation of your results and discuss them with the entire class next week.
▪ Submit Python code to create visualizations via Moodle until Friday midnight.

TODO:
1) Exploratory script (first point)
   - Create script and put important info into the presentation
   - Plots
   - ECG and PPG quality
   - Which signal segments were removed due to quality
   - Create a function for reading ECG signals, ready to be used in task 3
   - ...
2) Splitting script (second point)
   - Reproducible script so all group members get identical results
   - Describe the splitting pattern in the presentation
3) ECG Pan-Tompkins algorithm (third point)
   - Implementation of the algorithm for QRS peak detection
   - Method for extracting HR from detected peaks, preferably with variable segment length so we can adapt it for our algorithm's objective
   - Option to build an HR database for this segment length for later algorithm validation and training
   - Slides with a sneak peek of how the algorithm works on our data and the HR extraction method
4) Literature review (fourth point)
   - Review a few types of algorithms and select the one we will use
   - Describe it in the presentation
5) Presentation (fifth point)
