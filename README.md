# Pioneer# exporting the model
when training a new pioneer model:

    python pioneer_train.py

When exporting checkpoints as an onnx model, change code as the following:
1. line 11:
     device = torch.device("cpu")
     #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

2. line 247:
     #def train(self, replay_buffer, batch_data ,batch_size,writer):
      { 
        ```
     #}

3. line 241:
     #out=action[0,ind].cpu().data.numpy().flatten()
     #out=torch.tensor(out)
     out=action[0,ind]

After completing the changes , select the appropriate checkpoints file and run it within the terminal

    python pionner.py

