import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_datasets as tfds


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(7*7*256, use_bias = False, input_shape = (100,))
        self.relu1 = tf.keras.layers.ReLU()
        
        self.reshape = tf.keras.layers.Reshape((7,7,256))
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, (5,5), strides = (1,1), padding = 'same', use_bias = False)
        self.relu2 = tf.keras.layers.ReLU()
        
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, (5,5), strides = (2,2), padding = 'same', use_bias = False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        
        self.deconv3 = tf.keras.layers.Conv2DTranspose(1, (5,5), strides = (2,2), padding = 'same', use_bias = False, activation = 'tanh')
        
    def __call__(self, x, training = True):
        x = self.relu1(self.bn1(self.dense1(x)))
        x = self.reshape(x)
        x = self.relu2(self.bn2(self.deconv1(x)))
        x = self.relu3(self.bn3(self.deconv2(x)))
        x = self.deconv3(x)
        
        return x
        # keras에서 제공되는 DCGAN에 대한 문서와 논문을 바탕으로 만들었다. 우선 batchnormalization은 layer에 적용한다.
        # 논문 속에서 activation function은 tanh를 사용하는 마지막 layer를 제외하곤 모두 relu였다.
        # generator는 fractional-strided를 사용한다고 나오는데 이를 가능케 하는 것이 Conv2DTranspose이다.
                          
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5,5), strides = (2,2), padding='same', input_shape = [28,28,1])
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        self.conv2 = tf.keras.layers.Conv2D(128, (5,5), strides = (2,2), padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky2 = tf.keras.layers.LeakyReLU(alpha = 0.2)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation = 'sigmoid')
    
    def __call__(self,x,training = True):
        x = self.leaky1(self.bn1(self.conv1(x)))
        x = self.leaky2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dense(x)
        
        return x
        
        # 역시 batchnormalizaion을 적용했다. 또한 discriminator의 경우, activation function을 leakyrelu를 사용한다. 이 때, slope는 0.2이다. 
        # Pooling layer 대신 strided convolution을 사용했다.
    

def discriminator_loss(loss_object, real_output, fake_output):
    #here = tf.ones_like(????) or tf.zeros_like(????)  -> tf.zeros_like와 tf.ones_like에서 선택하고 (???)채워주세요
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    # real_loss의 경우, real_output과 같은 사이즈의 1로 이루어진 tensor를 만들어야 한다. 실제라면 모두 1의 값이 나올 것이기 때문이다. 그리고 이를 real_output과 비교하여 loss를 구한다.
    # faek_loss의 경우, fake_output과 같은 사이즈의 0으로 이루어진 tensor를 만들어야 한다. 가짜라면 모두 0의 값이 나올 것이기 때문이다. 그리고 이를 fake_output과 비교하여 loss를 구한다.
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output),fake_output)
    # generator는 real_output과 비슷한 결과값을 도출해야한다. 그러므로 fake_output과 같은 사이즈의 1로 이루어진 tensor(실제라면 1이라고 나올거니까)를 만든 fake_output과 비교하여 loss를 구해야 한다.
    
def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image


def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='/data/tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100 # 논문 속 figure1을 보게되면 Z(latent_vector)의 차원을 100으로 지정했다.
    epochs = 2
    batch_size = 10000
    buffer_size = 6000
    save_interval = 1

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1 = 0.5, beta_2 = 0.999)
    disc_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1 = 0.5, beta_2 = 0.999)
    # 논문 속 구조에 따르면 optimizer는 Adam을 사용했다.
    # 0.001은 너무 빨라서 0.0002로 대체했으며 beta_1은 학습의 안정화를 위해 0.5로 대체했다.
    # beta_2는 별다른 언급이 없으므로 디폴트인 0.999로 설정한다.
    
    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    # 진짜인지 아닌지를 바탕으로 확률값을 구하는 것이므로 binarycrossentropy를 사용한다.
    # 모델의 출력값이 sigmoid를 거쳐 만들어지므로 from_logits = True로 설정한다. 
    # 자세한 내용은 페이지 참조 : https://hwiyong.tistory.com/335
    
    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise)
            # genrated_images는 Z가 generator를 통해 noise와 섞여서 나온 결과여야 한다.
            
            real_output = discriminator(images)
            generated_output = discriminator(generated_images)
            # real_output은 입력할 image데이터를 discriminator에 넣은 값이다. 
            # generated_output은 generator가 만든 generated_image를 discriminator에 넣은 값이다.
            # 즉, discriminator에 input을 image와 generated_image로 받아 결과값을 도출하는 과정.
                  
            gen_loss = generator_loss(cross_entropy, generated_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)
            # generator와 discriminator는 따로 학습이 진행되어야 하므로 loss를 따로 구해줘야 한다.
            # 위에서 만든 각각의 loss_function에 사용할 cross_entropy(위에서 설정해둠)와 비교 대상을 넣는다.
            # gen_loss의 경우, generated_output(fake)를 넣고, discriminator는 real_output과 generated_output을 넣어준다.
            
        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()
                  
                  
# Reference 
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://simpling.tistory.com/entry/DCGANDeep-Convolutional-GAN