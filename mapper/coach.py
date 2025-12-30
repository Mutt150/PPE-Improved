import os
import paddle
import paddle.optimizer as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from latent_mappers import LevelsMapper
from stylegan import StyleGANv2Generator

class Coach:
    def __init__(self, opts):
        self.opts = opts
        
        # --- 1. Initialize Networks ---
        self.mapper = LevelsMapper()
        self.decoder = StyleGANv2Generator(1024, 512, 8)
        
        # Load Decoder (StyleGAN)
        if os.path.exists(self.opts.decoder_path):
            self.decoder.set_state_dict(paddle.load(self.opts.decoder_path))
        else:
            print(f"Warning: Decoder path {self.opts.decoder_path} not found.")
        self.decoder.eval()
        
        # Load Mapper
        if self.opts.mapper_path and os.path.exists(self.opts.mapper_path):
            print(f"Loading mapper from {self.opts.mapper_path}...")
            self.mapper.set_state_dict(paddle.load(self.opts.mapper_path))
        else:
            print("Initializing new mapper (random weights)...")
            
        # --- 2. Config Training ---
        self.optimizer = optim.Adam(parameters=self.mapper.parameters(), learning_rate=0.002)

        # Load Data
        if os.path.exists(self.opts.data_path):
            self.data = np.load(self.opts.data_path)
        else:
            print(f"Warning: Data path {self.opts.data_path} not found.")
            self.data = []

    def train(self, max_steps=200):
        """
        Fungsi Training dengan Strategi 'Smart Constraints'
        (Digunakan jika kamu menjalankan train_improvement.py)
        """
        self.mapper.train()
        print(f"Starting Training with Layer-wise Constraints...")
        print("Strategy: High penalty on Face Shape (Gender), Low penalty on Hair.")
        
        loss_history = []
        
        for step in range(max_steps):
            if len(self.data) == 0: break
            
            idx = np.random.randint(0, len(self.data))
            w = paddle.to_tensor(self.data[idx:idx+1])
            
            # Forward Pass
            delta = self.mapper(w)
            
            # --- LOGIKA TRAINING ---
            # 1. Structural Loss (Layer 0-4): KUNCI MATI
            loss_structure = paddle.mean(paddle.abs(delta[:, :4, :])) * 10.0
            
            # 2. Hair/Style Loss (Layer 4-8): BIARKAN BERUBAH
            loss_hair = paddle.mean(paddle.abs(delta[:, 4:8, :])) * 0.01
            
            # 3. Fine Loss (Layer 8+): Regularisasi standar
            loss_fine = paddle.mean(paddle.abs(delta[:, 8:, :])) * 0.5
            
            # Total Loss
            loss_total = loss_structure + loss_hair + loss_fine
            
            # Backward Pass
            self.optimizer.clear_grad()
            loss_total.backward()
            self.optimizer.step()
            
            loss_history.append(loss_total.item())
            
            if step % 20 == 0:
                print(f"Step {step} | Gender Loss: {loss_structure.item():.4f} | Hair Cost: {loss_hair.item():.4f}")

        # Simpan Model Baru
        save_path = os.path.join(self.opts.exp_dir, 'mapper_trained.pdparams')
        os.makedirs(self.opts.exp_dir, exist_ok=True)
        paddle.save(self.mapper.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def validate(self, eF=1.0, save=True):
        """
        Fungsi Validasi dengan 'ULTRA-STRICT MASKING' + 2 GRAFIK ANALISIS
        """
        self.mapper.eval()
        print(f"Starting validation with Ultra-Strict Masking & Full Analysis...")
        print("Strategy: Muting Layers 0-4 (Gender) and Layers 8+ (Makeup).")
        
        limit = min(len(self.data), 10) 
        
        stats = {
            'indices': [], 
            # Data Grafik 1 (Layer Analysis)
            'structure_change': [], 
            'hair_change': [],      
            'fine_change': [],
            # Data Grafik 2 (Sparsity L1/L2)
            'l1_sparsity': [],
            'l2_magnitude': []
        }

        for idx in range(limit):
            print(f"Processing image {idx+1}/{limit}...", end='\r')
            w = paddle.to_tensor(self.data[idx:idx+1])

            with paddle.no_grad():
                # 1. Generate Gambar Asli
                x, _ = self.decoder([w], input_is_latent=True, randomize_noise=True, truncation=1)
                
                # 2. Hitung Prediksi Perubahan (Delta)
                delta = self.mapper(w)
                
                # --- [INTI SOLUSI: LAYER MASKING] ---
                # A. Kunci Gender & Bentuk Wajah (Coarse Layers 0-4)
                delta[:, :4, :] = 0 

                # B. Kunci Makeup & Warna Kulit (Fine Layers 8+)
                delta[:, 8:, :] = 0
                
                # Hanya Layer 4-8 (Middle) yang tersisa.
                # ------------------------------------

                # Terapkan perubahan
                delta_w = 0.1 * delta * eF
                w_hat = w + delta_w
                
                # 3. Generate Gambar Edit
                x_hat, _ = self.decoder([w_hat], input_is_latent=True, randomize_noise=True, truncation=1)

                # 4. Statistik untuk Grafik
                # A. Statistik Layer
                struct_change = paddle.mean(paddle.abs(delta_w[:, :4, :])).item()
                hair_change = paddle.mean(paddle.abs(delta_w[:, 4:8, :])).item()
                fine_change = paddle.mean(paddle.abs(delta_w[:, 8:, :])).item()
                
                # B. Statistik Sparsity
                l1_val = paddle.mean(paddle.abs(delta_w)).item()
                l2_val = paddle.norm(delta_w, p=2).item()
                
                stats['indices'].append(idx)
                stats['structure_change'].append(struct_change)
                stats['hair_change'].append(hair_change)
                stats['fine_change'].append(fine_change)
                stats['l1_sparsity'].append(l1_val)
                stats['l2_magnitude'].append(l2_val)

            if save:
                self.parse_and_log_images(x, x_hat, title='images_ultra_strict', index=idx, eF=eF)
        
        print("\nProcessing complete. Generating ALL graphs...")
        # Panggil fungsi plotting baru yang menangani 2 grafik
        self.plot_all_graphs(stats)

    def parse_and_log_images(self, x, x_hat, title, index=None, eF=1.0):
        """Fungsi simpan gambar dengan perbaikan format uint8"""
        x_out = paddle.concat([x, x_hat], axis=3)
        x_out = paddle.transpose(x_out[0], [1, 2, 0]) * 0.5 + 0.5
        x_out = x_out.numpy()[:, :, ::-1]
        
        x_out = np.clip(x_out * 255, 0, 255).astype(np.uint8)
        
        path = os.path.join(self.opts.exp_dir, title, f'res_{index}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, x_out)

    def plot_all_graphs(self, stats):
        """Membuat 2 Grafik: Layer Analysis & Sparsity Analysis"""
        indices = stats['indices']
        
        # --- GRAFIK 1: Layer Disentanglement (Bukti Sukses Gender Aman) ---
        plt.figure(figsize=(10, 6))
        plt.plot(indices, stats['hair_change'], label='Hair (Target)', color='green', marker='o', linewidth=2)
        plt.plot(indices, stats['structure_change'], label='Gender (Locked)', color='red', marker='x', linestyle='--')
        plt.plot(indices, stats['fine_change'], label='Makeup (Locked)', color='blue', marker='.', linestyle=':')
        
        plt.title("Improved Model: Layer Disentanglement Analysis")
        plt.xlabel("Image Index")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path1 = os.path.join(self.opts.exp_dir, 'improved_layer_analysis.png')
        plt.savefig(save_path1)
        plt.close()

        # --- GRAFIK 2: Sparsity L1/L2 (Bukti Efisiensi) ---
        plt.figure(figsize=(12, 6))
        
        # Subplot L1
        plt.subplot(1, 2, 1)
        plt.plot(indices, stats['l1_sparsity'], marker='o', color='royalblue', linewidth=2)
        plt.title(f'Improved Model: Sparsity (L1)\n(Should be LOWER/STABLE)', fontsize=11)
        plt.xlabel('Image Index')
        plt.ylabel('|Δw|')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Subplot L2
        plt.subplot(1, 2, 2)
        plt.bar(indices, stats['l2_magnitude'], color='orange', alpha=0.7)
        plt.title(f'Improved Model: Magnitude (L2)', fontsize=11)
        plt.xlabel('Image Index')
        plt.ylabel('Euclidean')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path2 = os.path.join(self.opts.exp_dir, 'improved_sparsity_analysis.png')
        plt.savefig(save_path2)
        plt.close()
        
        print(f"Graphs saved:\n1. {save_path1}\n2. {save_path2}")