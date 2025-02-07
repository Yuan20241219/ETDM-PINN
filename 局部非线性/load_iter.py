import torch
import numpy as np
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray


class SeqDataIter:
    def __init__(self, data, batch_size, num_step,*,mean=None,var=None):
        self.num_step = num_step
        self.data,self.mean,self.var = self._data_init(data,mean,var)
        self.batch_size = batch_size
        self.len_data = self.data.shape[0]
        self.current_batch = 0
        self.current_step = 0



    def _data_init(self,data,mean,var):
        data = torch.from_numpy(np.array(data)).float()
        if len(data.shape)==1:
            data = data.unsqueeze(0)
        if mean is None:
            mean = data.mean()
            var = data.var()
        data=(data-mean)/torch.sqrt(var)
        data_mul = torch.stack(([torch.zeros(data.shape)] * self.num_step), dim=2)
        data=torch.cat((torch.zeros(data.shape[0],self.num_step-1),data), dim=1)
        for i in range(data_mul.shape[1]):
            data_mul[:, i, :] = data[:, i:i + self.num_step].flip(dims=[1])

        return data_mul,mean,var


    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch * self.batch_size >= self.len_data:
            self.current_batch = 0
            raise StopIteration

        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        batch_data = self.data[start:end, :,:]
        self.current_batch += 1

        return batch_data


if __name__ == '__main__':
    data = np.random.rand(180,1500)
    batch_size = 60
    num_step = 30
    iterator = SeqDataIter(data, batch_size, num_step)

    i=0
    for batch in iterator:
        i+=1
        print(i)
        print(batch)  # 打印每个批次的形状
