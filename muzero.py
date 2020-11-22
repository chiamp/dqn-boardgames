
class MuZeroModel:
    def __init__(self):
        # observation input , hidden state output
        representation_layers = [tf.keras.Input(shape=(4,))] + [ tf.keras.layers.Dense(4, activation=tf.nn.relu) for _ in range(2) ] + [tf.keras.layers.Dense(5, activation=tf.nn.relu)]
        current_layer = representation_layers[0]
        for i in range(1,len(representation_layers)): current_layer = representation_layers[i]( current_layer )
        self.representation_function = tf.keras.Model(inputs=representation_layers[0], outputs=current_layer) # h(o) --> s

        # hidden state + action input , (hidden state,reward) output
        dynamics_layers = [tf.keras.Input(shape=(5,)),tf.keras.Input(shape=(1,))] + [ tf.keras.layers.Dense(5, activation=tf.nn.relu) for _ in range(2) ] + [ tf.keras.layers.Dense(5, activation=tf.nn.relu) , tf.keras.layers.Dense(1, activation=tf.nn.tanh) ]
        current_layer = tf.keras.layers.concatenate( [ dynamics_layers[0] , dynamics_layers[1] ] )
        for i in range(2,len(dynamics_layers)-2): current_layer = dynamics_layers[i]( current_layer )
        dynamics_hidden_state_output = dynamics_layers[-2]( current_layer )
        dynamics_reward_output = dynamics_layers[-1]( current_layer )
        self.dynamics_function = tf.keras.Model(inputs=[dynamics_layers[0],dynamics_layers[1]], outputs=[dynamics_hidden_state_output,dynamics_reward_output]) # g(s,a) --> s',r

        # hidden_state input , # (policy,value) output
        prediction_layers = [tf.keras.Input(shape=(5,))] + [ tf.keras.layers.Dense(5, activation=tf.nn.relu) for _ in range(2) ] + [ tf.keras.layers.Dense(2, activation=tf.nn.softmax) , tf.keras.layers.Dense(1, activation=tf.nn.tanh) ]
        current_layer = prediction_layers[0]
        for i in range(1,len(prediction_layers)-2): current_layer = prediction_layers[i]( current_layer )
        prediction_policy_output = prediction_layers[-2]( current_layer )
        prediction_value_output = prediction_layers[-1]( current_layer )
        self.prediction_function = tf.keras.Model(inputs=prediction_layers[0], outputs=[prediction_policy_output,prediction_value_output]) # f(s) --> p,v

        self.trainable_variables = self.representation_function.trainable_variables + self.dynamics_function.trainable_variables + self.prediction_function.trainable_variables

        self.num_training_steps = 0
