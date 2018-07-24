import numpy as np
import tensorflow as tf
import time
import math
import helper
import os

#train_file = "data/dbpedia.train"
#test_file = "data/dbpedia.test"
train_file = "../dataset/haha_new/haha_new.raw.20180712.train"
test_file = "../dataset/haha_new/haha_new.raw.20180712.test"

# Number of Epochs
num_epochs = 10
max_window_size = 100
batch_size = 300
learning_rate_value = 0.01
show_every_n_batches = 10
log_dir = 'logs/'

label_to_int,vocab_to_int,int_to_label,int_to_vocab = helper.build_vocab_dict(train_file)
print ('label_to_int:', len(label_to_int))
print ('vocab_to_int:', len(vocab_to_int))

train_data = helper.read_data_into_cache(train_file)
test_data = helper.read_data_into_cache(test_file)

def get_batches(start_pos, batch_size, train_data):
    batch = train_data[start_pos:start_pos + batch_size]
    x = np.zeros((batch_size, max_window_size))
    mask = np.zeros((batch_size, max_window_size))
    y = []
    word_num = np.zeros((batch_size))
    line_no = 0
    col_no_stat = {}
    for line in batch:
        line = line.strip().lower().split(' ')
        col_no = 0
        for i in line:
            if i in label_to_int:
                y.append(label_to_int[i])
        for i in line:
            if i in vocab_to_int:
                x[line_no][col_no] = vocab_to_int[i]
                mask[line_no][col_no] = 1
                col_no += 1
            if col_no >= max_window_size:
                break
        word_num[line_no] = col_no
        line_no += 1
        if col_no not in col_no_stat:
            col_no_stat[col_no] = 0
        col_no_stat[col_no] += 1

    y = np.array(y).reshape(batch_size, 1)
    batch_mask = mask.reshape(batch_size, max_window_size, 1)
    word_number = word_num.reshape(batch_size, 1)
    #print('col_no_stat:', col_no_stat)
    return x,y,batch_mask,word_number

def get_inputs():
    inputs = tf.placeholder(tf.int32, shape=[None, max_window_size])
    targets = tf.placeholder(tf.int64, [None, 1])
    vocab_num = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, targets, learning_rate, vocab_num

vocab_size = len(vocab_to_int)+1
embed_dim = 128
def get_embed(input_data, vocab_size, embed_dim):
    emb_mask = tf.placeholder(tf.float32, shape=[None, max_window_size, 1])
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    project_embedding = tf.div(tf.reduce_sum(tf.multiply(embed, emb_mask), 1),vocab_size)
    return embed,project_embedding,emb_mask


def build_relu_layer(x,width):
    weights = tf.Variable(tf.random_normal([embed_dim, width]))
    biases = tf.Variable(tf.random_normal([width]))
    layer = tf.add(tf.matmul(x, weights), biases)
    layer = tf.nn.relu(layer)
    return layer

n_classes = len(label_to_int)
relu_layer_width = 128

train_graph = tf.Graph()
with train_graph.as_default():
    inputs, targets, learning_rate,vocab_num = get_inputs()
    embed,project_embedding,emb_mask = get_embed(inputs, vocab_size, embed_dim)
    # Construct model
    pred = build_relu_layer(project_embedding,relu_layer_width)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([n_classes, relu_layer_width],
                stddev=1.0 / math.sqrt(relu_layer_width)))

    nce_biases = tf.Variable(tf.zeros([n_classes]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=targets,
                     inputs=pred,
                     num_sampled=10,
                     num_classes=n_classes))
    cost = tf.reduce_sum(loss) / batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases

with tf.Session(graph=train_graph) as sess:
    #Initialize Saver
    saver=tf.train.Saver()
    # Open a writer to write summaries.
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('cost', cost)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    if os.path.exists(log_dir+"checkpoint"):
        print("Restoring Variables from Checkpoint.")
        saver.restore(sess,tf.train.latest_checkpoint(log_dir))
    else:
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())

    # Training cycle
    start_time = time.time()
    total_batch = int(len(vocab_to_int) / batch_size)
    print("total_batch of training data: ", total_batch)
    save_path=log_dir+"model.ckpt"
    for epoch in range(num_epochs):  # TODO should incre epoch if use saver.restore
        avg_cost = 0.
        for i in range(total_batch):
            x, y, batch_mask, word_number = get_batches(i * batch_size, batch_size, train_data)
            summary,_,c = sess.run([merged, optimizer, cost], feed_dict={inputs: x, targets: y,
                                                         emb_mask: batch_mask, vocab_num: word_number,
                                                         learning_rate:learning_rate_value})
            avg_cost += c / total_batch
            writer.add_summary(summary, epoch)
        # Display logs per epoch step
        if epoch % show_every_n_batches == 0:
            end_time = time.time()
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost), "time=", "{:.4f}".format(end_time-start_time))
            start_time = time.time()
            saver.save(sess,save_path,global_step=epoch)

    saver.save(sess,save_path,global_step=epoch)
    writer.close()
    
    fout = open(log_dir+'preds.txt', 'w')
    fout.write('title+body\ttrue\tpred\n')
    # Test model
    prediction = tf.argmax(out_layer, 1)
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.reshape(targets, [batch_size]))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    total_batch = int(len(test_data) / batch_size)
    final_accuracy = 0
    for i in range(total_batch):
        x, y, batch_mask, word_number = get_batches(i*batch_size, batch_size, test_data)
        batch_accuracy = accuracy.eval({inputs: x, targets: y,
                                        emb_mask: batch_mask, vocab_num: word_number,
                                        learning_rate: learning_rate_value})
        preds = prediction.eval({inputs: x, targets: y,
                                 emb_mask: batch_mask, vocab_num: word_number,
                                 learning_rate: learning_rate_value})
        #print('preds:', preds)
        for j in range(len(x)):
            xj = []
            for k in range(len(x[j])):
                if int(x[j][k]) > 0:
                    xj.append(int_to_vocab[int(x[j][k])])
            fout.write(' '.join(xj)+'\t'+int_to_label[int(y[j])]+'\t'+int_to_label[preds[j]]+'\n')
        print("Batch Accuracy: ", batch_accuracy)
        final_accuracy += batch_accuracy
    print("Final Accuracy: ", final_accuracy * 1.0 / total_batch)
    fout.close()

