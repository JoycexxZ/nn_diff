import torch
import torch.nn as nn


class Latent_AE_cnn_big(nn.Module):
    def __init__(
            self,
            in_dim,
            channel=6,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 4
        self.kernal_size = 4
        self.channel_list = [channel, channel, channel, channel]
        self.channel_list_dec = [8, 256, 256, channel]
        self.real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=0),
            nn.Tanh(),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.ConvTranspose1d(
                self.channel_list_dec[3], self.channel_list_dec[3], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list_dec[3], self.channel_list_dec[2], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.ConvTranspose1d(
                self.channel_list_dec[2], self.channel_list_dec[2], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list_dec[2], self.channel_list_dec[1], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.ConvTranspose1d(
                self.channel_list_dec[1], self.channel_list_dec[1], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list_dec[1], self.channel_list_dec[0], self.kernal_size, stride=1,
                      padding=self.fold_rate - 1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.ConvTranspose1d(
                self.channel_list_dec[0], self.channel_list_dec[0], self.kernal_size, stride=self.fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list_dec[0], 1, self.kernal_size, stride=1, padding=self.fold_rate),
        )

        # self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input):
        input_shape = input.shape
        # import pdb;pdb.set_trace()
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        emb_enc4 = emb_enc4 + torch.randn_like(emb_enc4) * 0.1

        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :input_shape[-1]]

        return emb_dec4.reshape(input_shape)

    def Enc(self, input):
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device),
            ],
            dim=2,
        )
        emb_enc1 = self.enc1(input)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        return emb_enc4

    def Dec(self, emb_enc4):
        emb_dec1 = self.dec1(emb_enc4)
        emb_dec2 = self.dec2(emb_dec1)
        emb_dec3 = self.dec3(emb_dec2)
        emb_dec4 = self.dec4(emb_dec3)[:, :, :self.in_dim]
        emb_dec4 = emb_dec4.reshape(emb_dec4.shape[0], -1)

        return emb_dec4