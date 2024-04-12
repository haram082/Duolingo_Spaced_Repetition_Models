from keras.layers import Input, Dense
from keras.models import Model 
import matplotlib.pyplot as plt
from hierarchical import scaled_features

encoding_dim = 32

input_data = Input(shape=(scaled_features.shape[1],))

encoded = Dense(encoding_dim, activation='relu')(input_data)

decoded = Dense(scaled_features.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)

encoder = Model(input_data, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=256, shuffle=True)

encoded_data = encoder.predict(scaled_features)


plt.scatter(encoded_data[:,0], encoded_data[:,1], alpha=0.5)
plt.title('Encoded data visualization')
plt.xlabel('Encoded feature 1')
plt.ylabel('Encoded feature 2')
plt.show()