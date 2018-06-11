"""Policy Gradient Reinforcement Learning with Tensorflow"""

import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import gym

from collections import deque

# Open AI gym Atari env: 0: 'NOOP', 2: 'UP', 3: 'DOWN'
ACTIONS = [0, 2, 3]

OBSERVATION_DIM = 80 * 80

MEMORY_CAPACITY = 100000
ROLLOUT_SIZE = 10000

# MEMORY stores tuples:
# (observation, label, reward)
MEMORY = deque([], maxlen=MEMORY_CAPACITY)


def gen():
    for m in list(MEMORY):
        yield m


# helpers taken from:
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r.tolist()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-directory', default='/tmp/pong')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--train-batch-size', default=10000, type=int)
    parser.add_argument('--save-checkpoint-steps', default=10, type=int)
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--decay', type=float, default=0.99)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('model'):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, input_shape=(OBSERVATION_DIM, ),
                                  use_bias=False, activation='relu', name='hidden'),
            tf.keras.layers.Dense(len(ACTIONS), use_bias=False, name='logits')
        ])

    with tf.name_scope('rollout'):
        observations = tf.placeholder(shape=(None, OBSERVATION_DIM), dtype=tf.float32)
        logits = model(observations, training=False)
        sample_action = tf.squeeze(tf.multinomial(logits=logits, num_samples=1))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=args.decay)

    with tf.name_scope('dataset'):
        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=(tf.float32, tf.int32, tf.float32))
        dataset = dataset.shuffle(MEMORY_CAPACITY).repeat(None).batch(args.train_batch_size)
        iterator = dataset.make_one_shot_iterator()

    with tf.name_scope('train'):
        train_observations, labels, processed_rewards = iterator.get_next()

        # This reuses the same weights in the rollout phase.
        train_observations.set_shape((args.train_batch_size, OBSERVATION_DIM))
        train_logits = model(train_observations)

        cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_logits,
            labels=labels)

        loss = tf.reduce_sum(processed_rewards * cross_entropies)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    with tf.name_scope('summaries'):
        rollout_reward = tf.placeholder(shape=(), dtype=tf.float32)

        tf.summary.scalar('rollout_reward', rollout_reward)
        tf.summary.scalar('loss', loss)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        merged = tf.summary.merge_all()

    env = gym.make('Pong-v0')
    previous_x = None
    _rollout_reward = -21.0

    with tf.Session() as sess:
        restore_path = tf.train.latest_checkpoint(args.model_directory)
        if restore_path:
            saver.restore(sess, restore_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        summary_path = os.path.join(args.model_directory, 'summary')
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        for epoch in range(args.n_epoch):
            tf.logging.info('Starting epoch {}'.format(epoch))
            _observation = env.reset()

            epoch_memory = []
            episode_memory = []

            while True:
                current_x = prepro(_observation)
                diff_x = current_x - previous_x if previous_x is not None else np.zeros(OBSERVATION_DIM)
                previous_x = current_x

                _label = sess.run(sample_action, feed_dict={observations: [diff_x]})
                _action = ACTIONS[_label]
                _observation, _reward, _done, _ = env.step(_action)

                if args.render: env.render()

                # record experience
                episode_memory.append((diff_x, _label, _reward))

                if _done:
                    obs, lbl, rwd = zip(*episode_memory)

                    # processed rewards
                    prwd = discount_rewards(rwd, args.gamma)
                    prwd -= np.mean(prwd)
                    prwd /= np.std(prwd)

                    # store the processed experience to memory
                    epoch_memory.extend(zip(obs, lbl, prwd))

                    # calculate the running rollout reward
                    _rollout_reward = 0.9 * _rollout_reward + 0.1 * sum(rwd)

                    episode_memory = []

                    _observation = env.reset()
                    previous_x = None

                if len(epoch_memory) >= ROLLOUT_SIZE:
                    break

            # add to the global memory
            MEMORY.extend(epoch_memory)

            tf.logging.info('Starting training')
            tf.logging.info('Rollout reward: {}'.format(_rollout_reward))

            # Here we train only once.
            _, _global_step = sess.run([train_op, global_step])

            if _global_step % args.save_checkpoint_steps == 0:
                tf.logging.info('Writing summary')
                summary = sess.run(merged, feed_dict={rollout_reward: _rollout_reward})
                summary_writer.add_summary(summary, _global_step)

                save_path = os.path.join(args.model_directory, 'model.ckpt')
                save_path = saver.save(sess, save_path, global_step=_global_step)
                tf.logging.info('Model checkpoint saved: {}'.format(save_path))
