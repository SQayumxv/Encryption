import wx
import ChaosAlgorithm
from cryptography.fernet import Fernet
import hashlib
import logging
from StatisticalAnalysis import calculate_entropy, chi_squared_test, calculate_compression_ratio, \
    calculate_kl_divergence, calculate_hamming_distance, calculate_frequency_distribution
from StatisticalAnalysis import plot_histogram
from StatisticalAnalysis import calculate_correlation_coefficient
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas




logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("encryption.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class FileGUI(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title)


        # Create the main sizer for layout
        self.SetMinSize((600, 400))
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)  # Use a vertical sizer

        self.SetBackgroundColour(wx.Colour(27, 28, 24))
        # File Path Label
        self.file_path_label = wx.StaticText(self, label="File Path:")
        self.file_path_label.SetForegroundColour(wx.Colour(227,228,221))
        self.main_sizer.Add(self.file_path_label, 0, wx.ALL | wx.ALIGN_LEFT, 5)  # Add with padding

        self.file_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.file_path_ctrl = wx.TextCtrl(self)
        self.browse_button = wx.Button(self, label="Browse...")
        self.browse_button.SetBackgroundColour(wx.Colour(43,76,74))
        self.browse_button.SetForegroundColour(wx.Colour(227,228,221))
        self.browse_button.SetWindowStyleFlag(wx.BORDER_NONE)

        self.file_path_sizer.Add(self.file_path_ctrl, 3, wx.ALL| wx.EXPAND, 5)  # Add with padding and expand
        self.file_path_sizer.Add(self.browse_button, 0, wx.ALL, 5)  # Add with padding and center alignment

        self.main_sizer.Add(self.file_path_sizer, 0, wx.ALL | wx.EXPAND, 5)

        self.key_label = wx.StaticText(self, label="Encryption Key:")
        self.key_label.SetForegroundColour(wx.Colour(227,228,221))

        self.main_sizer.Add(self.key_label, 0, wx.ALL | wx.ALIGN_LEFT, 5)  # Add with padding

        self.key_input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.key_ctrl = wx.TextCtrl(self, style=wx.TE_PASSWORD)
        self.generate_key_button = wx.Button(self, label="Generate Key")
        self.copy_key_button = wx.Button(self, label="Copy Key")

        # self.dummy_btn = wx.Button(self, label="dummy...")
        self.key_input_sizer.Add(self.key_ctrl, 3, wx.ALL | wx.EXPAND, 5)  # Add with padding and expand
        self.key_input_sizer.Add(self.generate_key_button, 0, wx.ALL, 5)  # Add with padding and center alignment
        self.key_input_sizer.Add(self.copy_key_button, 0, wx.ALL, 5)  # Add with padding and center alignment

        self.main_sizer.Add(self.key_input_sizer, 0, wx.ALL | wx.EXPAND, 5)

        self.encrypt_button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Encrypt Button
        self.encrypt_button = wx.Button(self, label="Encrypt")
        self.decryption_button = wx.Button(self, label="Decrypt")

        self.encrypt_button_sizer.Add(self.encrypt_button, 0, wx.ALL | wx.ALIGN_CENTER,
                            5)  # Add with padding and center alignment
        self.encrypt_button_sizer.Add(self.decryption_button, 0, wx.ALL | wx.ALIGN_CENTER,
                            5)  # Add with padding and center alignment
        self.main_sizer.Add(self.encrypt_button_sizer, 0, wx.ALL | wx.CENTER, 5)

        # Optional: Result TextCtrl (for displaying encrypted data)
        # self.result_ctrl = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        # self.main_sizer.Add(self.result_ctrl, 0, wx.ALL | wx.EXPAND, 5)  # Add with padding and expand

        # Set the frame sizer (to manage element layout)

        self.SetSizer(self.main_sizer)

        # Bind events (button clicks)
        self.Bind(wx.EVT_BUTTON, self.on_browse_clicked, self.browse_button)
        self.Bind(wx.EVT_BUTTON, self.on_encrypt_clicked, self.encrypt_button)
        self.Bind(wx.EVT_BUTTON, self.on_decrypt_clicked, self.decryption_button)
        self.Bind(wx.EVT_BUTTON, self.on_generate_key_clicked, self.generate_key_button)
        self.Bind(wx.EVT_BUTTON, self.on_copy_key_clicked, self.copy_key_button)


    def on_browse_clicked(self, event):
        """Handles the click event of the "Browse..." button"""
        with wx.FileDialog(self, "Select a file", wildcard="*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                # Get the selected file path
                file_path = fileDialog.GetPath()
                self.file_path_ctrl.SetValue(file_path)

    def on_encrypt_clicked(self, event):
        file_path = self.file_path_ctrl.GetValue()
        key = self.key_ctrl.GetValue()
        Block_Size = 32
        try:
            with open(file_path, 'rb') as file:
                plaintext = file.read()

            ciphertext = ChaosAlgorithm.encrypt(file_path, key, Block_Size)
            if ciphertext:
                min_length = min(len(plaintext), len(ciphertext))
                adjusted_plaintext = plaintext[:min_length]
                adjusted_ciphertext = ciphertext[:min_length]

                byte_array = bytearray(adjusted_ciphertext)
                entropy_value = calculate_entropy(byte_array)
                logger.info(f"Entropy of ciphertext: {entropy_value}")

                histogram_filename = "ciphertext_histogram.png"
                plot_histogram(byte_array, filename=histogram_filename)

                correlation = calculate_correlation_coefficient(
                    bytearray(adjusted_plaintext), byte_array
                )
                logger.info(f"Correlation coefficient: {correlation}")

                # Perform Chi-Squared test
                chi_stat, p_value = chi_squared_test(byte_array)
                logger.info(f"Chi-squared test stat: {chi_stat}, p-value: {p_value}")

                # Hamming Distance
                hamming_distance = calculate_hamming_distance(adjusted_plaintext, adjusted_ciphertext)
                logger.info(f"Hamming distance: {hamming_distance}")

                # Compression Ratio
                compression_ratio = calculate_compression_ratio(byte_array)
                logger.info(f"Compression ratio: {compression_ratio}")

                distribution_from_plaintext = calculate_frequency_distribution(adjusted_plaintext)
                distribution_from_ciphertext = calculate_frequency_distribution(adjusted_ciphertext)

                # Ensure distributions cover the entire byte range (0-255)
                full_range_distribution_plaintext = [distribution_from_plaintext.get(i, 0) for i in range(256)]
                full_range_distribution_ciphertext = [distribution_from_ciphertext.get(i, 0) for i in range(256)]

                # Add a small number (like 1e-10) to all counts to avoid division by zero when converting counts to probabilities
                full_range_distribution_plaintext = [x + 1e-10 for x in full_range_distribution_plaintext]
                full_range_distribution_ciphertext = [x + 1e-10 for x in full_range_distribution_ciphertext]

                # Normalize distributions
                sum_plaintext = sum(full_range_distribution_plaintext)
                sum_ciphertext = sum(full_range_distribution_ciphertext)
                normalized_distribution_plaintext = [x / sum_plaintext for x in full_range_distribution_plaintext]
                normalized_distribution_ciphertext = [x / sum_ciphertext for x in full_range_distribution_ciphertext]

                # KL Divergence
                kl_divergence = calculate_kl_divergence(normalized_distribution_plaintext, normalized_distribution_ciphertext)
                logger.info(f"KL Divergence: {kl_divergence}")


                encrypted_image_filename = file_path.replace(".jpg", "_encrypted.png").replace(".png", "_encrypted.png").replace(".gif", "_encrypted.gif")
                ChaosAlgorithm.visualize_encrypted_data(ciphertext, encrypted_image_filename)

                wx.MessageBox(
                    f"File encrypted successfully!\nEntropy: {entropy_value}\nHistogram saved as {histogram_filename}\nCorrelation coefficient: {correlation}\nChi-squared test stat: {chi_stat}, p-value: {p_value}\nHamming distance: {hamming_distance}\nCompression ratio: {compression_ratio}\nKL Divergence: {kl_divergence}.",
                    "Success", wx.ICON_INFORMATION)
            else:
                logger.error("Encryption failed, no ciphertext returned.")
                wx.MessageBox("Encryption failed!", "Error", wx.ICON_ERROR)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            wx.MessageBox(f"Error during encryption: {e}", "Error", wx.ICON_ERROR)

    def on_decrypt_clicked(self, event):
        # Get the selected file path and the encryption key from the GUI
        file_path = self.file_path_ctrl.GetValue()
        key = self.key_ctrl.GetValue()

        try:
            decrypted_data = ChaosAlgorithm.decrypt(file_path, key)

            # If decryption was successful
            if decrypted_data:
                # Here you might want to save the decrypted data to a file
                # or update the GUI to reflect the successful decryption.
                wx.MessageBox("File decrypted successfully!", "Success", wx.ICON_INFORMATION)

                # If you want to calculate entropy for the decrypted data,
                # you can call the calculate_entropy function here.
                # entropy_value = calculate_entropy(decrypted_data)
                # logger.info(f"Entropy of decrypted data: {entropy_value}")

            else:
                logger.error("Decryption Success, decrypted data returned.")
                wx.MessageBox("Decryption Success!", "Success", wx.ICON_ERROR)

        except Exception:
            logger.error(f"Decryption failed: ")
            wx.MessageBox(f"Error during decryption: ", "Error", wx.ICON_ERROR)

    def on_generate_key_clicked(self, event):
        """Handles the click event of the "Generate Key" button"""
        file_path = self.file_path_ctrl.GetValue()
        
        try:
            with open(file_path, 'rb') as file:
                binary_data = file.read()
                bytearray_str = "bytearray(b\"" + "".join("\\x{:02x}".format(byte) for byte in binary_data) + "\")"
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            return None
        hashed_value = hashlib.sha256(bytearray_str.encode('utf-8')).digest()
        key = Fernet.generate_key()
        combined_key = key + bytes(hashed_value)
        self.key_ctrl.SetValue(key.decode())  # Display the decoded key for user

    def on_copy_key_clicked(self, event):
        """Handles the click event of the "Copy Key" button"""
        key_text = self.key_ctrl.GetValue()
        if key_text:
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(key_text))
                wx.TheClipboard.Close()
                wx.MessageBox("Key copied to clipboard!", "Success", wx.ICON_INFORMATION)

if __name__ == "__main__":
    app = wx.App()
    frm = FileGUI(None, title="FileEncryptor")
    frm.Show()
    app.MainLoop()
