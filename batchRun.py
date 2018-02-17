num_epochs = 1
batch_size = 20

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001) 
converter_optimizer = torch.optim.Adam(converter.parameters(), lr=0.0001) 
criterion = nn.CrossEntropyLoss()

#len(train_id_to_file)

for _ in range(num_epochs):
    total_loss = 0
    for i in range(50000//batch_size):
        
        start_idx = i * batch_size % len(train_id_to_file)
        
        batch_sen = [[] for y in range(batch_size*5)] 
    
        count=0
        for train_id in train_ids[start_idx:start_idx + batch_size]:
            for n in range(5):
                batch_sen[count+(n*batch_size)].append(train_id_to_captions[train_id][n])
            count+=1    
        
        for cap in range(5):
            
            set_batch = []
            for batch in batch_sen[(cap*batch_size):(cap+1)*batch_size]: 
                set_batch.append(batch[0])
        
            training_input,training_output,w2v_input,one_hot_output,input_lens = create_training(
                start_idx, start_idx + batch_size,set_batch)
        
            loss = train(training_input,
                     training_output, 
                     w2v_input,
                     one_hot_output,
                     encoder,
                     decoder, 
                     decoder_optimizer,
                     converter_optimizer,
                     input_lens,    
                     criterion)
            
            total_loss+=loss
        
        
        if i % 50 == 0:
            print(i,total_loss/5000)
            total_loss = 0
    

torch.save(encoder.state_dict(), './encoder4.pth')
torch.save(converter.state_dict(), './converter4.pth')
torch.save(decoder.state_dict(), './decoder4.pth')
print("training done") 