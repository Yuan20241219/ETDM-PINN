from structure import Structure
import torch


class ETDM_structure():
    def __init__(self,L, structure, dt, n_steps,*, beta=0.25, gamma=0.5):
        self.dof = structure.dof
        self.L = L
        self.n_steps=n_steps
        self.structure = structure
        self.a0 = 1 / (beta * dt ** 2)
        self.a1 = 1 / (beta * dt)
        self.a2 = 1 / (2 * beta) - 1
        self.a3 = gamma / (beta * dt)
        self.a4 = gamma / beta - 1
        self.a5 = (gamma / beta - 2) * dt / 2
        self.M=torch.tensor(structure.M).float()
        self.K0=torch.tensor(structure.K).float()
        self.C=torch.tensor(structure.C).float()

        self.A=self._get_A()


    def _get_H(self, dof, M, K0, C):
        H = torch.zeros((dof * 2, dof * 2))
        H[dof:, :dof] = -torch.inverse(M) @ K0
        H[:dof, dof:] = torch.eye(dof)
        H[dof:, dof:] = -torch.inverse(M) @ C
        return H

    def _get_W(self, dof, M):
        W = torch.zeros((dof * 2, self.L.shape[1]))
        W[dof:] = torch.inverse(M) @ self.L
        return W

    def _get_S1(self, dof):
        S1 = torch.zeros((dof * 2, dof * 2))
        S1[:dof, :dof] = self.a3 * torch.eye(dof)
        S1[dof:, :dof] = self.a0 * torch.eye(dof)
        return S1

    def _get_S2(self, dof):
        S2 = torch.zeros((dof * 2, dof * 2))
        S2[:dof, :dof] = self.a4 * torch.eye(dof)
        S2[dof:, :dof] = self.a1 * torch.eye(dof)
        S2[:dof, dof:] = self.a5 * torch.eye(dof)
        S2[dof:, dof:] = self.a2 * torch.eye(dof)
        return S2


    def _get_A(self):
        H = self._get_H(self.dof, self.M, self.K0, self.C)
        W = self._get_W(self.dof, self.M)
        S1 = self._get_S1(self.dof)
        S2 = self._get_S2(self.dof)

        Q1 = -torch.inverse((H - S1)) @ S2 @ W
        Q2 = -torch.inverse((H - S1)) @ W
        T = -torch.inverse(H - S1) @ (S1 + S2 @ H)

        A1_1 = Q2
        A2_1 = T @ Q2 + Q1
        A = torch.stack((A1_1, A2_1))
        for i in range(2, self.n_steps):
            A = torch.cat((A, torch.unsqueeze(T @ A[-1], 0)), dim=0)
        A = torch.flip(A, [0])

        return A


if __name__ == '__main__':
    L = torch.zeros((5, 2))
    L[:, 0] = 5000
    L[:, 1] = -torch.tensor(([1, 0, 0, 0, 0]))

    structure=Structure(dof=5,M=[5000]*5,K=[1e7]*5,num_omega1=1,num_omega2=2,zeta=0.05)

    structure=ETDM_structure(L,structure,dt=0.02,n_steps=600)

    print(structure.A.shape)

