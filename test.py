import encode_data

data = encode_data.one_hot_encode("/home/rkdesai7/Documents/Kozac_Decoder/real_kozac.txt","/home/rkdesai7/Documents/Kozac_Decoder/fake_kozac.txt",)
print(data)
data = encode_data.extract_data("/home/rkdesai7/Documents/Kozac_Decoder/real_kozac.txt","/home/rkdesai7/Documents/Kozac_Decoder/fake_kozac.txt",)
print(data)

