"""Policy Gradient Reinforcement Learning with Tensorflow Eager"""

import numpy as np
import os
import gym
from agents.tools.wrappers import AutoReset, FrameHistory
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from argparse import ArgumentParser
from collections import deque


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

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
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train-batch-size', default=10000, type=int)
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--decay', type=float, default=0.99)
    args = parser.parse_args()

    inner_env = gym.make('Pong-v0')
    env = FrameHistory(inner_env, past_indices=[0, 1], flatten=False)
    env = AutoReset(env)

    writer = tf.contrib.summary.create_file_writer(args.model_directory)
    writer.set_as_default()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, input_shape=(OBSERVATION_DIM, ), use_bias=False,
                              activation='relu', name='hidden'),
        tf.keras.layers.Dense(len(ACTIONS), use_bias=False, name='logits')
    ])

    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=args.decay)

    global_step = tf.train.get_or_create_global_step()

    # checkpoint
    checkpoint_prefix = os.path.join(args.model_directory, "model.ckpt")
    root = tfe.Checkpoint(
        optimizer=optimizer,
        model=model,
        optimizer_step=global_step)

    root.restore(tf.train.latest_checkpoint(args.model_directory))

    # create dataset
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
    dataset = dataset.shuffle(MEMORY_CAPACITY).repeat()
    dataset = dataset.batch(args.train_batch_size)

    _rollout_reward = -21.0

    for i in range(args.epochs):
        episode_memory = []
        epoch_memory = []

        print('>>> Rollout phase')

        _observation = np.zeros(OBSERVATION_DIM)

        while True:
            logits = model(tf.constant([_observation]), training=False)
            _label = tf.squeeze(tf.multinomial(logits=logits, num_samples=1)).numpy()

            action = ACTIONS[_label]

            _pair_state, _reward, _done, _ = env.step(action)

            # record experience
            episode_memory.append((_observation, _label, _reward))

            current_obs, previous_obs = _pair_state

            prepro_current_obs = prepro(current_obs)
            prepro_previous_obs = prepro(previous_obs)

            _observation = prepro_current_obs - prepro_previous_obs

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

                # reset episode memory
                episode_memory = []

            if len(epoch_memory) >= ROLLOUT_SIZE:
                break

        # add to the global memory
        MEMORY.extend(epoch_memory)

        print('>>> Train phase')
        print('rollout reward: {}'.format(_rollout_reward))

        # start training
        train_observations, labels, processed_rewards = tfe.Iterator(dataset).next()

        with tf.GradientTape() as tape:
            train_logits = model(train_observations)
            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=train_logits)
            loss = tf.reduce_sum(processed_rewards * cross_entropies)

        print('Loss at step {}: {:.3f}'.format(global_step.numpy(), loss))

        grads = tape.gradient(loss, model.variables)

        optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)

        root.save(file_prefix=checkpoint_prefix)

        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.scalar('loss', loss)
            tf.contrib.summary.scalar('rollout_reward', _rollout_reward)
            tf.contrib.summary.histogram('hidden', model.get_layer('hidden').kernel)
            tf.contrib.summary.histogram('logits', model.get_layer('logits').kernel)
