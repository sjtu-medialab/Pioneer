import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import onnxruntime as ort

#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
b_min = torch.tensor([10000.]).to(device)
b_max = torch.tensor([8000000.]).to(device)
sub_weights = torch.tensor([738496.125,694683.3125,691479.25,673617.375,666556.0625,681600.1875,669463.3125,
                666129,662238.1875,659593.75,7.3224053382873535,6.640115737915039,6.661839008331299,6.499343395233154,
                6.463366985321045,66.41350555419922,65.2419204711914,65.0151138305664,64.62989807128906,64.40997314453125,5549.7080078125,
                5218.357421875,5197.064453125,5061.17431640625,5007.40673828125,51043.18359375,50241.6328125,50011.48046875,49687.8984375,
                49473.1640625,51.766231536865234,43.154056549072266,46.73100662231445,44.35700988769531,45.82548522949219,54.172786712646484,
                52.25625228881836,52.80331802368164,52.15654754638672,51.818397521972656,-8.076197624206543,-9.032713890075684,-8.184115409851074,
                -8.881279945373535,-8.57718276977539,-6.027364253997803,-7.842897415161133,-7.138024806976318,-7.600769519805908,-7.766489505767822,
                151.33204650878906,151.2038116455078,151.07054138183594,150.9722900390625,150.88592529296875,151.33204650878906,150.38157653808594,149.42373657226562,
                148.60403442382812,147.83462524414062,1.048581838607788,0.9403055310249329,0.965387225151062,0.9410541653633118,0.9549955725669861,1.2477848529815674,
                1.222879409790039,1.2160511016845703,1.2063682079315186,1.194684624671936,12.129443168640137,9.123194694519043,9.858774185180664,9.137080192565918,
                9.637124061584473,28.483427047729492,27.244096755981445,27.210426330566406,27.071775436401367,26.76516342163086,20.601516723632812,17.87394142150879,
                18.152664184570312,18.502660751342773,17.923423767089844,18.974851608276367,18.55058479309082,18.663055419921875,18.46375274658203,
                18.452112197875977,8.178370475769043,5.5424723625183105,6.423187732696533,5.719115734100342,6.158593654632568,12.5791015625,11.946151733398438,
                12.197224617004395,11.995094299316406,12.022625923156738,0.03417285904288292,0.029939087107777596,0.0304565466940403,0.030316563323140144,
                0.030231211334466934,0.04919557273387909,0.0460132472217083,0.04782531410455704,0.04682319611310959,0.04794519394636154,0.31053441762924194,
                0.27375665307044983,0.27528613805770874,0.271941214799881,0.26977038383483887,0.8403489589691162,0.7605294585227966,0.8009557127952576,0.771939218044281,
                0.8093596696853638,0.5885034799575806,0.5472908020019531,0.561969518661499,0.555245578289032,0.5488224029541016,0.5749943256378174,
                0.5683791637420654,0.5661110281944275,0.5616078972816467,0.5590522885322571,0.42020806670188904,0.41124776005744934,0.40748050808906555,0.40527647733688354,
                0.414554625749588,0.43730807304382324,0.43566998839378357,0.4336211681365967,0.4324624836444855,0.43000659346580505,0.004606906324625015,0.004368430934846401,
                0.004239862319082022,0.00433636549860239,0.004258331842720509,0.005296767223626375,0.005273005925118923,0.005282784346491098,0.00526299886405468,0.005225192755460739
        # ... include all the provided weights here ...
            ], dtype=torch.float32)
mul_weights = torch.tensor([ 9.108388780987298e-7,0.000001059008923220972,0.0000010491108923815773,0.0000011062785461035674,0.0000011233828445256222,0.0000012176325299151358,
                0.0000012276935876798234,0.000001225727373821428,0.0000012282000625418732,0.0000012310398460613214,0.11547315120697021,0.1370280683040619,0.13245660066604614,0.14291255176067352,
                0.14601102471351624,0.01579149439930916,0.015916554257273674,0.015866877511143684,0.015860475599765778,0.01589406281709671,0.00012113105913158506,0.00014051565085537732,0.00013927537656854838,
                0.00014692841796204448,0.0001496588665759191,0.000016243957361439243,0.000016617701476207003,0.00001658736618992407,0.000016620524547761306,0.00001669244738877751,0.007275158539414406,
                0.009120622649788857,0.008939903229475021,0.009114483371376991,0.008997362107038498,0.007180711720138788,0.00841580517590046,0.007690922822803259,0.008248683996498585,0.008288963697850704,0.00620803888887167,
                0.007471443619579077,0.007342279423028231,0.0074492571875452995,0.007365750148892403,0.006154336035251617,0.006908596493303776,0.006505969446152449,0.006853444501757622,0.006897413171827793,
                0.008673721924424171,0.008673866279423237,0.008672562427818775,0.008672185242176056,0.008671551011502743,0.008673721924424171,0.008665213361382484,0.00865672342479229,0.008649756200611591,
                0.008646947331726551,1.799630045890808,2.278015613555908,2.185457706451416,2.502725124359131,2.1366732120513916,0.7276115417480469,0.7529850602149963,0.7551690936088562,0.7569636106491089,0.7683706283569336,0.04213936626911163,
                0.07190877199172974,0.06961502134799957,0.07673759758472443,0.07813231647014618,0.021535275503993034,0.026588434353470802,0.025132175534963608,0.02553657442331314,0.026560090482234955,0.036353643983602524,
                0.04710008576512337,0.048716530203819275,0.04568418487906456,0.048586416989564896,0.042830608785152435,0.052614402025938034,0.047247227281332016,0.0500192828476429,0.049893688410520554,0.06829073280096054,
                0.12170270830392838,0.1143275797367096,0.12081389874219894,0.11699365079402924,0.057100776582956314,0.07574842870235443,0.0641368180513382,0.07225460559129715,0.07336115837097168,8.385283470153809,
                8.79195785522461,8.734068870544434,8.727130889892578,8.733592987060547,9.911806106567383,10.059857368469238,9.928916931152344,9.957329750061035,9.947797775268555,0.5528498291969299,0.6015455722808838,
                0.6040346622467041,0.6128904819488525,0.6106728315353394,0.45432814955711365,0.4981219172477722,0.47951459884643555,0.4930354356765747,0.47873011231422424,3.129629373550415,2.868824005126953,2.8642611503601074,2.9201724529266357,
                2.926238536834717,3.7120745182037354,3.6531338691711426,3.61527681350708,3.5694897174835205,3.5351955890655518,3.234508514404297,2.8877975940704346,2.9905972480773926,2.9318313598632812,2.9699902534484863,
                3.759488582611084,3.7236077785491943,3.7136762142181396,3.6999588012695312,3.696364641189575,20.4775333404541,19.821626663208008,20.3739013671875,19.948535919189453,20.258892059326172,34.844886779785156,
                34.0961799621582,33.8519172668457,33.95980453491211,34.118324279785156
            ], dtype=torch.float32)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super().__init__()

        self.fc1 = nn.Linear(151, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.identity=nn.Identity()

        self.sub_weights = nn.Parameter(sub_weights.view(1, -1), requires_grad=False)
        self.mul_weights = nn.Parameter(mul_weights.view(1, -1), requires_grad=False)

        self.max_action = 1
        self.phi = phi

    def forward(self, state, action):

        state = (state-self.sub_weights) * self.mul_weights 
        action = (torch.log(action) - torch.log(b_min)) / (torch.log(b_max) - torch.log(b_min))
        action = (action - 0.5)*2
        action = torch.clamp(action, min=-1, max=1)

        sa = torch.cat([state, action], -1)

        a = F.relu(self.fc1(sa))
        a = F.relu(self.fc2(a))
        a = self.phi * self.max_action * torch.tanh(self.fc3(a))
        res = (a + action).clamp(-self.max_action, self.max_action) 
        res = res.squeeze(-1)
        return res


class Critic(nn.Module):
    def __init__(self, state_dim=150, action_dim=1):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 1)
        self.sub_weights = nn.Parameter(sub_weights.view(1, -1), requires_grad=False)
        self.mul_weights = nn.Parameter(mul_weights.view(1, -1), requires_grad=False)
      


    def forward(self, state, action):

        state = (state-self.sub_weights) * self.mul_weights 
        action = (torch.log(action) - torch.log(b_min)) / (torch.log(b_max) - torch.log(b_min))
        action = torch.clamp(action, min=-1, max=1)
        action = (action - 0.5)*2
        action = action.unsqueeze(-1)

        sa = torch.cat([state, action], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


    def q1(self, state, action):
        state = (state-self.sub_weights) * self.mul_weights 

        action = (torch.log(action) - torch.log(b_min)) / (torch.log(b_max) - torch.log(b_min))
        action = (action - 0.5)*2
        action = torch.clamp(action, min=-1, max=1)
        action = action.unsqueeze(-1)

        sa = torch.cat([state, action], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 512)
        self.e2 = nn.Linear(512, 512)

        self.mean = nn.Linear(512,latent_dim)
        self.log_std = nn.Linear(512, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 512)
        self.d2 = nn.Linear(512, 512)
        self.d3 = nn.Linear(512, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device
        self.sub_weights = nn.Parameter(sub_weights.view(1, -1), requires_grad=False)
        self.mul_weights = nn.Parameter(mul_weights.view(1, -1), requires_grad=False)


    def forward(self, state, action):

        state_temp=state
        state = (state-self.sub_weights) * self.mul_weights
        action = (torch.log(action) - torch.log(b_min)) / (torch.log(b_max) - torch.log(b_min))
        action = (action - 0.5)*2

        action = torch.clamp(action, min=-1, max=1)
        action = action.unsqueeze(-1)

        sa = torch.cat([state, action], -1)
        z = F.relu(self.e1(sa))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state_temp, z)

        return u, mean, std


    def decode(self, state, z=None):

        state = (state-self.sub_weights) * self.mul_weights 

        if z is None:
            z = torch.randn((state.shape[0], state.shape[1],self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
        

class Pioneer(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        super().__init__()
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-8) #-8

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.counter = 0
        self.identity=nn.Identity()


    def forward(self, state , h, c):		
        with torch.no_grad():

            h=self.identity(h)
            c=self.identity(c)
            state=state.squeeze(0)
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            state=state.unsqueeze(0)

            k=self.vae.decode(state)
            k=(k+1)/2
            k = torch.exp(k * (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min))) + torch.log(torch.tensor(b_min)))
            action = self.actor(state, k)

            action=(action+1)/2
            action = torch.exp(action * (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min))) + torch.log(torch.tensor(b_min)))
            
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(1)
            
            out=action[0,ind].cpu().data.numpy().flatten()
            out=torch.tensor(out)
            #out=action[0,ind]
        return out, h, c
    

    def train(self, replay_buffer, batch_data ,batch_size,writer):
        # Sample replay buffer / batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_data)
 
        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(state, action)

        b_max = 8000000
        b_min =   10000
    
        action_vae = (torch.log(action) - torch.log(torch.tensor(b_min))) / (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min)))
        action_vae = (action_vae- 0.5)*2
        recon_loss = F.mse_loss(recon, action_vae.unsqueeze(-1))

        writer.add_scalar(' Training Loss -recon', recon_loss,self.counter)
        
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        
        writer.add_scalar(' Training Loss -vae', vae_loss,self.counter)
        self.counter +=1
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        h = torch.zeros(1, 256).to(device)
        c = torch.zeros(1, 256).to(device)

        # Critic Training
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)

            # Compute value of perturbed  actions sampled from the VAE

            new_action=self.vae.decode(next_state)
            new_action=(new_action+1)/2
            new_action = torch.exp(new_action* (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min))) + torch.log(torch.tensor(b_min)))
            
            critic_action=self.actor_target(next_state, new_action)
            critic_action=(critic_action+1)/2
            critic_action = torch.exp(critic_action* (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min))) + torch.log(torch.tensor(b_min)))

            target_Q1, target_Q2 = self.critic_target(next_state, critic_action)

            # Soft Clipped Double Q-learning 
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)

            sat = None
            for i in range(int(target_Q.shape[0]/10)):
                tmp = target_Q[i*10:i*10+10,:,:]
                res = tmp.permute(2,1,0)
                res=res.max(2)[0]
                if sat is None:
                    sat=res
                else:
                    sat=torch.cat([sat,res],0)
            target_Q = sat.unsqueeze(-1)
            reward = reward.unsqueeze(-1)
            target_Q = reward + not_done * self.discount * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        writer.add_scalar(' Training Loss -critic_step', critic_loss ,self.counter)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state)
        sampled_actions = (sampled_actions+1)/2
        sampled_actions = torch.exp(sampled_actions * (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min))) + torch.log(torch.tensor(b_min)))
        
        perturbed_actions = self.actor(state, sampled_actions)


        action_loss= (torch.log(action) - torch.log(torch.tensor(b_min))) / (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min)))
        action_loss= (action_loss - 0.5)*2

        actor_action_loss=perturbed_actions 

        perturbed_actions = (perturbed_actions+1)/2
        perturbed_actions = torch.exp(perturbed_actions  * (torch.log(torch.tensor(b_max)) - torch.log(torch.tensor(b_min))) + torch.log(torch.tensor(b_min)))
    
        # Update through DPG
        actor_loss = -self.critic.q1(state, perturbed_actions).mean() + F.mse_loss(actor_action_loss,action_loss)
  
        writer.add_scalar(' Training Loss -actor_step', actor_loss ,self.counter)
        writer.add_scalar(' Training Loss -actor_step_q1', -self.critic.q1(state, perturbed_actions).mean() ,self.counter)
        writer.add_scalar(' Training Loss -actor_step_BC', F.mse_loss(actor_action_loss,action_loss) ,self.counter)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Update Target Networks 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss,actor_loss,reward
    
    def save(self, filename):
        torch.save(self.state_dict(), filename + "_Pioneer")
        

        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.vae.state_dict(), filename + "_vae")
        torch.save(self.vae_optimizer.state_dict(), filename + "_vae_optimizer")


    def load(self, filename):
        self.load_state_dict(torch.load(filename + "_Pioneer"))
        
if __name__ == "__main__":
    # batch size
    BS = 1
    # time steps
    T = 1
    # observation vector dimension
    obs_dim = 150
    # number of hidden units in the LSTM
    hidden_size = 128
    act_dim = 1
    kwargs = {
        "state_dim": obs_dim,
        "action_dim": act_dim,
        "max_action": 1,
        "device" : device,
        "discount": 0.99,
        "tau": 0.005,
        "lmbda": 0.75,
        "phi"  : 0.05,

    }
    torchBwModel = Pioneer(**kwargs)
    # update the model
    checkpoint = torch.load('./Pioneer/checkpoints/checkpoint_epoch_120_Pioneer')
    torchBwModel.load_state_dict(checkpoint)

    # create dummy inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = np.asarray(np.random.uniform(0, 1, size=(BS, T, obs_dim)), dtype=np.float32)
    torch_dummy_inputs = torch.as_tensor(dummy_inputs)
    torch_initial_hidden_state = torch.zeros((BS, hidden_size))
    torch_initial_cell_state = torch.zeros((BS, hidden_size))

    # predict dummy outputs: 1 episode x T timesteps x 2 (mean and std)
    dummy_outputs, final_hidden_state, final_cell_state = torchBwModel(torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state)
    print(dummy_outputs)
    
    # save onnx model
    model_path = "./onnx_model/pioneer_final.onnx"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torchBwModel.to("cpu")
    torchBwModel.eval()


    torch.onnx.export(
        torchBwModel,
        (torch_dummy_inputs[0:1, 0:1, :], torch_initial_hidden_state, torch_initial_cell_state),
        model_path,
        opset_version=11,
        input_names=['obs', 'hidden_states', 'cell_states'], # the model's input names
        output_names=['output', 'state_out', 'cell_out'], # the model's output names
        verbose=True,

    )

    # verify tf and onnx models outputs
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    onnx_hidden_state, onnx_cell_state = (np.zeros((1, hidden_size), dtype=np.float32), np.zeros((1, hidden_size), dtype=np.float32))
    torch_hidden_state, torch_cell_state = (torch.as_tensor(onnx_hidden_state), torch.as_tensor(onnx_cell_state))
    # online interaction: step through the environment 1 time step at a time
    with torch.no_grad():
        for i in tqdm(range(dummy_inputs.shape[1])):
            torch_estimate, torch_hidden_state, torch_cell_state = torchBwModel(torch_dummy_inputs[0:1, i:i+1, :], torch_hidden_state, torch_cell_state)
            feed_dict= {'obs': dummy_inputs[0:1,i:i+1,:], 'hidden_states': onnx_hidden_state, 'cell_states': onnx_cell_state}
            onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
            assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=1e-3), 'Failed to match model outputs!'
            assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-6), 'Failed to match hidden state1'
            assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-6), 'Failed to match cell state!'

        assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-6), 'Failed to match final hidden state!'
        assert np.allclose(torch_cell_state, final_cell_state, atol=1e-6), 'Failed to match final cell state!'
        print("Torch and Onnx models outputs have been verified successfully!")