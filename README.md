# plants_vs_anomalies
Environmental Hackathon Submission


Project Pitch:


Plants versus Anomalies uses a deep learning model trained on Amazon SageMaker and deployed on an Nvidia Jetson Nano to find anomalies in data streamed to it from environmental sensors in Amazon's Spheres in Seattle. The horticulturalist uses our easy webpage to set a threshold value. If streamed data exceeds the threshold an alert is texted to the horticulturalist's cellphone. If one of the carbon dioxide sensors is the cause of the anomaly, the Jetson Nano triggers a special light to turn on above the plants in that area that will open their pores and empower them to suck excessive co2 from the environment. The light turns off afterwards to allow the plant to rest. The horticulturalist can see graphs of both streamed data and historical data, including anomalies found, on the webpage. 



Further Project Description:

A fairly lean model - a two hidden layer autoencoder - without using temporal data seemed to do a reasonable job of finding anomalies. We inserted fake anomalies into the data and the model found them. 
Building an LSTM into the autoencoder did better for us




Project Stacks, APIs or other Technology Used:


Models deployed on an Nvidia Jetson Nano

Models trained using Amazon SageMaker running on a GPU instance

Model is an AutoEncoder built using the Keras library of TensorFlow

The webpage uses a Dash component to render the graphs. 




