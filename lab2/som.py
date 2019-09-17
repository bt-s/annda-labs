import numpy as np

class GetData():

    def get_animal_data(self,file):
        file = open(file,'r')
        props = np.zeros((32,84))
        for line in file:
            attributes = line.split(',')
            props = np.asarray(attributes).reshape((32,84))
        return props.astype(int)

    def get_animal_names(self,file):
        file = open(file,'r')
        names = []
        for line in file:
            name = line.split()
            names.append(name[0])
        return np.asarray(names).reshape(-1,1)

    def get_cities(self,file):
        pass

    def get_votes(self,file):
        pass

data = GetData()

props = data.get_animal_data("data_lab2/animals.dat") #32x84
animal_names = data.get_animal_names('data_lab2/animalnames.txt') #32x1


class Som():

    def __init__(self, x, hidden_nodes=, input_nodes=84, output_nodes=100, epochs=20, eta=0.2, nHood_size=50, neighbourhood="1D"):
        self.hidden_nodes = hidden_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.epochs = epochs
        self.eta = eta
        #self.nHoodSize = nHoodSize
        self.neighbourhood = neighbourhood

    def weight_initialization(self):

        self.W = np.random.uniform(0,1,(100,84))
        return self.W

    def similarity(self,x,w):
        np.abs(np.sum(x - w))
        return np.dot( (x-w).T, (x-w))


    def learn(self, x, props):
        for e in range(n.epochs):
            nHoodsize = (50-e)

            for i in range(props[0]):
                animal_row = props[:,i]
                similarities ={}

                for index,w_row in self.W:
                    similarity = self.similarity(animal_row,w_row)
                    similarities[index] = similarity

                winner_index = max(similarities, key=similarities.get)
                index_range = (winner_index-nHoodsize, winner_index+nHoodsize)
                for index in range(winner_index):
                    neighboor_weights =  W[index]    #weights to be updated















