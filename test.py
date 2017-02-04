from keras.models import load_model
from scipy.ndimage import imread


model = load_model('model.h5')

print('Start')
image_array = imread('./data/IMG/center_2016_12_01_13_36_16_767.jpg')
transformed_image_array = image_array[None, :, 1:-1, :]
transformed_image_array = ((transformed_image_array / 255.) - 0.5) * 2
steering_angle = float(model.predict(transformed_image_array, batch_size=1))
print(steering_angle)
