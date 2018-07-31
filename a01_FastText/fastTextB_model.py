import tensorflow as tf
import numpy as np


class FastTextB(object):
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate, num_sampled, sentence_len,
                 vocab_size, embed_size, is_training):
        self.label_size = label_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_sampled = num_sampled
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.sentence_len = sentence_len
        self.is_training = is_training

        self.sentence = tf.placeholder(tf.int32, [None, sentence_len], name="sentence")  # [None, sentence_len]
        self.labels = tf.placeholder(tf.int32, [None,], name="Labels")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.init_weights()
        self.logits = self.inference()
        if not self.is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        corrections = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32), name="Accuracy")


    def init_weights(self):
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)  # [None, sentence_len, embed_size]
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1,
                                                  name="sentence_embeddings")  # [None, embed_size]
        return tf.matmul(self.sentence_embeddings, self.W) + self.b

    def loss(self, l2_lambda=0.01):
        if self.is_training:  # train
            labels = tf.reshape(self.labels, [-1])  # [batch_size, 1] ---> [batch_size,]
            labels = tf.expand_dims(labels, 1)  # [batch_size,] ---> [batch_size, 1]
            loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=tf.transpose(self.W),
                biases=self.b,
                labels=labels,
                inputs=self.sentence_embeddings,
                num_sampled=self.num_sampled,
                num_classes=self.label_size,
                partition_strategy="div"
            ))
        else:  # eval/inference
            labels_one_hot = tf.one_hot(self.labels, self.label_size)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            print("loss:0", loss)  # [None, label_size]
            loss = tf.reduce_sum(loss, axis=1)
            print("loss:0", loss)  # [None,]
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * l2_lambda
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate,
                                                   staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate,
                                                   optimizer="Adam")
        return train_op


def test():
    num_classes = 19
    batch_size = 8
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1
    fastText = FastTextB(num_classes, learning_rate, batch_size, decay_steps, decay_rate, 5, sequence_length,
                         vocab_size, embed_size, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sequence_length), dtype=np.int32)
            input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1], dtype=np.int32)
            loss, acc, predict, _ = sess.run(
                [fastText.loss_val, fastText.accuracy, fastText.predictions, fastText.train_op],
                feed_dict={fastText.sentence: input_x, fastText.labels: input_y})
            print("loss:", loss, "acc:", acc, "label", input_y, "prediction:", predict)


test()
print("ended...")
