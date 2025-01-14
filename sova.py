import cosmos_sdk
from cosmos_sdk import CosmosEnvironment, DroneAgent
import torch
from transformers import AutoTokenizer, AutoModel
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


env = CosmosEnvironment(config='base/cosmos_config.yaml')


model_name = 'nvidia/Cosmos-1.0-Diffusion-7B-Text2World'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


drone = DroneAgent(env, model=model, tokenizer=tokenizer)

def train_drone(drone, env, epochs=10):
    for epoch in range(epochs):
        print(f"Iniciando a época {epoch + 1}/{epochs}")
        
        
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            
            observation = state['observation']
            
            
            action = drone.generate_action(observation)
            
           
            next_state, reward, done, info = env.step(action)
            
            
            total_reward += reward
            
          
            state = next_state
            
            
            env.render()
        
        print(f"Época {epoch + 1} concluída com recompensa total: {total_reward}")
        
       
        drone.save_model(f'drone_model_epoch_{epoch + 1}.pth')

class DroneAgent:
    def __init__(self, env, model, tokenizer):
        self.env = env
        self.model = model
        self.tokenizer = tokenizer

    def generate_action(self, observation):
        
        input_text = observation['text']
        
        
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=50)
        
        
        action_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        action = self.parse_action(action_text)
        
        return action

    def parse_action(self, action_text):
        
        if "subir" in action_text.lower():
            return {'throttle': 1.0}
        elif "descer" in action_text.lower():
            return {'throttle': -1.0}
        elif "virar à esquerda" in action_text.lower():
            return {'yaw': -30}
        elif "virar à direita" in action_text.lower():
            return {'yaw': 30}
        else:
            return {'throttle': 0, 'yaw': 0}

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

if __name__ == "__main__":
    
    num_epochs = 20
    
    
    train_drone(drone, env, epochs=num_epochs)
    
    
    env.close()
