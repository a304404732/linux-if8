from Config import Config
import numpy as np
import torch
import torch.nn as nn
import transformers


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        config = transformers.GPT2Config(vocab_size=1, n_embd=args.hidden_size, n_layer=args.n_layer,
                                         n_inner=args.n_inner, n_head=args.n_head,
                                         activation_function=args.active_func,
                                         n_positions=args.n_position, resid_pdrop=args.res_pdrop,
                                         attn_pdrop=args.att_pdrop)
        self.transformer = transformers.GPT2Model(config)
        # embedding层
        self.embed_timestep = nn.Embedding(args.max_ep_len, args.hidden_size)
        self.embed_return = torch.nn.Linear(args.reward_dim, args.hidden_size)
        self.embed_state = torch.nn.Linear(args.state_dim, args.hidden_size)
        self.embed_action = torch.nn.Linear(args.action_dim, args.hidden_size)
        # layerNorm层
        self.embed_ln = nn.LayerNorm(args.hidden_size)
        # 输出层
        self.predict_state = torch.nn.Linear(args.hidden_size, args.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(args.hidden_size, args.action_dim)] + ([nn.Tanh()] if args.action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(args.hidden_size, args.reward_dim)
        self.max_length = args.sequence_len

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.args.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.args.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.args.state_dim)
        actions = actions.reshape(1, -1, self.args.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.args.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.args.action_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds[0, -1]


class Buffer:
    def __init__(self, args):
        self.args = args
        self.sub_memory = np.zeros([args.max_step, args.state_dim + args.action_dim + args.reward_dim])
        self.memory = np.zeros([args.buffer_size, args.max_step, args.state_dim + args.action_dim + args.reward_dim])
        self.sub_mem_ptr = 0
        self.mem_ptr = 0
        self.full_flag = 0

    def store_sub_memory(self, state, action, reward):
        transition = np.concatenate((np.array([reward]), state, action))
        self.sub_memory[self.sub_mem_ptr] = transition
        self.sub_mem_ptr += 1

    def clear_sub_memory(self):
        self.sub_mem_ptr = 0
        self.sub_memory = np.zeros([self.args.max_step, self.args.state_dim +
                                    self.args.action_dim + self.args.reward_dim])

    def update_memory(self):
        # 将子经验池的数据同步更新到总经验池中
        if self.mem_ptr >= self.args.buffer_size:
            self.mem_ptr = 0
            self.full_flag = 1
        self.memory[self.mem_ptr, :self.sub_memory.shape[0]] = self.sub_memory
        self.mem_ptr += 1

    def reward_to_go(self, x, gamma=1.0):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def sample(self):
        seq_len = self.args.sequence_len
        batch_size = self.args.batch_size
        # 从经验池中随机抽取回合的indices 可以抽中相同的回合，因为后续选序列长度的时候随机选
        batch_indices = np.random.choice(self.args.buffer_size, batch_size, replace=True)
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []
        for i in range(batch_size):
            # batch_indices是抽取回合的索引值，返回一个回合的经验二维tensor
            episode_memory = self.memory[batch_indices[i]]
            # episode_memory[:, 0]表示奖励
            si = np.random.randint(0, episode_memory[:, 0].shape[0] - 1)
            # 从经验池中获取数据
            s.append(episode_memory[:, 1:4][si:si + seq_len].reshape(1, -1, self.args.state_dim))
            a.append(episode_memory[:, 4][si:si + seq_len].reshape(1, -1, self.args.action_dim))
            r.append(episode_memory[:, 0][si:si + seq_len].reshape(1, -1, self.args.reward_dim))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= seq_len] = seq_len - 1  # 对padding 进行裁切

            rtg.append(self.reward_to_go(episode_memory[:, 0][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            # 给不够seq_len的状态、动作、奖励 打padding
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, seq_len - tlen, self.args.state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std  # 对一个batch的内容进行标准化
            a[-1] = np.concatenate([np.ones((1, seq_len - tlen, self.args.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, seq_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, seq_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, seq_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, seq_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.args.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.args.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.args.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.args.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.args.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.args.device)

        return s, a, r, rtg, timesteps, mask

    def sample_prior(self):
        states = 0
        actions = 0
        rewards = 0
        attention_mask = 0
        rtg = 0
        timesteps = 0

        return states, actions, rewards, rtg, timesteps, attention_mask

    def mem_is_full(self):
        return self.full_flag


class Agent:
    def __init__(self, args):
        self.args = args
        self.memory = Buffer(args)
        self.model = Transformer(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_func = torch.nn.MSELoss()

    def get_action(self, states, actions, rewards, rtg, timesteps):
        return self.model.get_action(states, actions, rewards, rtg, timesteps)

    def learn(self):
        train_loss = []
        self.model.train()  # 调用model的train方法时内部的LN层和dropout参数有效

        for _ in range(self.args.train_num):
            loss = self.learn_step()
            train_loss.append(loss)
        self.model.eval()  # 调用model的eval方法时，内部LN和drop参数固定
        return train_loss

    def learn_step(self):
        states, actions, rewards, rtg, timesteps, attention_mask = self.memory.sample()
        action_target = torch.clone(actions)
        _, action_preds, _ = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_func(action_preds, action_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)  # 梯度裁剪防止梯度更新过大
        self.optimizer.step()
        return loss.detach().cpu().item()
