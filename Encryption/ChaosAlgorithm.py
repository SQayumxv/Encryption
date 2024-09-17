import os
import hashlib
import numpy as np
import math

from matplotlib import pyplot as plt


def LogMapRNG(IntiralPapulation,GrowthRate,Pixles):
    RandomPix=np.zeros(Pixles)
    for t in range(0,Pixles):  
        IntiralPapulation=(GrowthRate*IntiralPapulation)*(1-IntiralPapulation)
        RandomPix[t]= math.ceil(IntiralPapulation*1000000000000000)
    return RandomPix

def ChoticNumberGenerator(Imagearr,key):
    ChaoticRNG=LogMapRNG(0.1,3.9,Imagearr)
    ChaoticRNG=ChaoticRNG%256
    numberofoiuyhn=2
    ChaoticRNG=ChaoticRNG/(256/numberofoiuyhn)
    for i in range (0,int(len(ChaoticRNG))):
        ChaoticRNG[i]=int(ChaoticRNG[i])
    return ChaoticRNG

def split_array(original_array, size):
    num_mini_arrays = (len(original_array) + size - 1) // size
    mini_arrays = [[] for _ in range(num_mini_arrays)]
    for i, item in enumerate(original_array):
        mini_arrays[i // size].append(item)
    return mini_arrays

oiuyhn = [[
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
],[
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]]
xcvtr = [[
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]
,[
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]
]




def sub_word(word: [int]) -> bytes:
    substituted_word = bytes(oiuyhn[0][i] for i in word)
    return substituted_word


def rcon(i: int) -> bytes:
    # From Wikipedia
    rcon_lookup = bytearray.fromhex('01020408102040801b36')
    rcon_value = bytes([rcon_lookup[i-1], 0, 0, 0])
    return rcon_value


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes([x ^ y for (x, y) in zip(a, b)])


def rot_word(word: [int]) -> [int]:
    return word[1:] + word[:1]


def key_expansion(key: bytes, nb: int = 4) -> [[[int]]]:

    nk = len(key) // 4

    key_bit_length = len(key) * 8

    if key_bit_length == 128:
        nr = 10
    elif key_bit_length == 192:
        nr = 12
    else:  # 256-bit keys
        nr = 14

    w = state_from_bytes(key)

    for i in range(nk, nb * (nr + 1)):
        temp = w[i-1]
        if i % nk == 0:
            temp = xor_bytes(sub_word(rot_word(temp)), rcon(i // nk))
        elif nk > 6 and i % nk == 4:
            temp = sub_word(temp)
        w.append(xor_bytes(w[i - nk], temp))

    return [w[i*4:(i+1)*4] for i in range(len(w) // 4)]


def add_round_key(state: [[int]], key_schedule: [[[int]]], round: int):
    round_key = key_schedule[round]
    for r in range(len(state)):
        state[r] = [state[r][c] ^ round_key[r][c] for c in range(len(state[0]))]





def sub_bytes(ChaoticPRN,state: [[int]]):
    #print(s_box)
    for r in range(len(state)):
        for c in range(len(state[0])):  
            state[r][c] = oiuyhn[int(ChaoticPRN[r*c])][state[r][c]]  

  
def shift_rows(state: [[int]]):
    if len(state) != 4 or any(len(row) != 4 for row in state):
        raise ValueError("Invalid state matrix dimensions for Nov shift rows.")
    
    # Perform row shifts as before
    state[0][1], state[1][1], state[2][1], state[3][1] = state[1][1], state[2][1], state[3][1], state[0][1]
    state[0][2], state[1][2], state[2][2], state[3][2] = state[2][2], state[3][2], state[0][2], state[1][2]
    state[0][3], state[1][3], state[2][3], state[3][3] = state[3][3], state[0][3], state[1][3], state[2][3]

def xtime(a: int) -> int:
    if a & 0x80:
        return ((a << 1) ^ 0x1b) & 0xff
    return a << 1


def mix_column(col: [int]):
    c_0 = col[0]
    all_xor = col[0] ^ col[1] ^ col[2] ^ col[3]
    col[0] ^= all_xor ^ xtime(col[0] ^ col[1])
    col[1] ^= all_xor ^ xtime(col[1] ^ col[2])
    col[2] ^= all_xor ^ xtime(col[2] ^ col[3])
    col[3] ^= all_xor ^ xtime(c_0 ^ col[3])


def mix_columns(state: [[int]]):
    for r in state:
        mix_column(r)



def state_from_bytes(data: bytes) -> [[int]]:
    state = []
    for i in range(0, len(data), 4):
        state.append(list(data[i:i+4]))
    return state


def bytes_from_state(state: [[int]]) -> bytes:
    return bytes(state[0] + state[1] + state[2] + state[3])



def Nov_encryption(ChaoticPRN,data: bytes, key: bytes) -> bytes:

    key_bit_length = len(key) * 8

    if key_bit_length == 128:
        nr = 10
    elif key_bit_length == 192:
        nr = 12
    else:  # 256-bit keys
        nr = 14

    state = state_from_bytes(data)
    key_schedule = key_expansion(key)

    add_round_key(state, key_schedule, round=0)

    for round in range(1, nr):
        sub_bytes(ChaoticPRN,state)
        shift_rows(state)
        mix_columns(state)
        add_round_key(state, key_schedule, round)

    sub_bytes(ChaoticPRN,state)
    shift_rows(state)
    add_round_key(state, key_schedule, round=nr)

    cipher = bytes_from_state(state)
    return cipher

def inv_shift_rows(state: [[int]]) -> [[int]]:
    # [00, 10, 20, 30]     [00, 10, 20, 30]
    # [01, 11, 21, 31] <-- [11, 21, 31, 01]
    # [02, 12, 22, 32]     [22, 32, 02, 12]
    # [03, 13, 23, 33]     [33, 03, 13, 23]
    state[1][1], state[2][1], state[3][1], state[0][1] = state[0][1], state[1][1], state[2][1], state[3][1]
    state[2][2], state[3][2], state[0][2], state[1][2] = state[0][2], state[1][2], state[2][2], state[3][2]
    state[3][3], state[0][3], state[1][3], state[2][3] = state[0][3], state[1][3], state[2][3], state[3][3]
    return



def inv_sub_bytes(ChaoticPRN, state: [[int]]) -> [[int]]:
    for r in range(len(state)):
        state[r] = [xcvtr[int(ChaoticPRN[r*c])][state[r][c]] for c in range(len(state[0]))]


def xtimes_0e(b):
    # 0x0e = 14 = b1110 = ((x * 2 + x) * 2 + x) * 2
    return xtime(xtime(xtime(b) ^ b) ^ b)


def xtimes_0b(b):
    # 0x0b = 11 = b1011 = ((x*2)*2+x)*2+x
    return xtime(xtime(xtime(b)) ^ b) ^ b


def xtimes_0d(b):
    # 0x0d = 13 = b1101 = ((x*2+x)*2)*2+x
    return xtime(xtime(xtime(b) ^ b)) ^ b


def xtimes_09(b):
    # 0x09 = 9  = b1001 = ((x*2)*2)*2+x
    return xtime(xtime(xtime(b))) ^ b


def inv_mix_column(col: [int]):
    c_0, c_1, c_2, c_3 = col[0], col[1], col[2], col[3]
    col[0] = xtimes_0e(c_0) ^ xtimes_0b(c_1) ^ xtimes_0d(c_2) ^ xtimes_09(c_3)
    col[1] = xtimes_09(c_0) ^ xtimes_0e(c_1) ^ xtimes_0b(c_2) ^ xtimes_0d(c_3)
    col[2] = xtimes_0d(c_0) ^ xtimes_09(c_1) ^ xtimes_0e(c_2) ^ xtimes_0b(c_3)
    col[3] = xtimes_0b(c_0) ^ xtimes_0d(c_1) ^ xtimes_09(c_2) ^ xtimes_0e(c_3)


def inv_mix_columns(state: [[int]]) -> [[int]]:
    for r in state:
        inv_mix_column(r)


def inv_mix_column_optimized(col: [int]):
    u = xtime(xtime(col[0] ^ col[2]))
    v = xtime(xtime(col[1] ^ col[3]))
    col[0] ^= u
    col[1] ^= v
    col[2] ^= u
    col[3] ^= v


def inv_mix_columns_optimized(state: [[int]]) -> [[int]]:
    for r in state:
        inv_mix_column_optimized(r)
    mix_columns(state)


def Nov_decryption(ChaoticPRN, cipher: bytes, key: bytes) -> bytes:

    key_byte_length = len(key)
    key_bit_length = key_byte_length * 8

    if key_bit_length == 128:
        nr = 10
    elif key_bit_length == 192:
        nr = 12
    else:  # 256-bit keys
        nr = 14

    state = state_from_bytes(cipher)
    key_schedule = key_expansion(key)
    add_round_key(state, key_schedule, round=nr)

    for round in range(nr-1, 0, -1):
        inv_shift_rows(state)
        inv_sub_bytes(ChaoticPRN, state)
        add_round_key(state, key_schedule, round)
        inv_mix_columns(state)

    inv_shift_rows(state)
    inv_sub_bytes(ChaoticPRN, state)
    add_round_key(state, key_schedule, round=0)

    plain = bytes_from_state(state)
    return plain



Nov_BLOCK_SIZE = 16


def Nov_Blocks_encryption(plain: bytes, key: bytes) -> bytes:
    # Assumption: length of data is multiple of 128 bits
    cipher = []
    block=(len(plain)+(len(plain)%8)) // Nov_BLOCK_SIZE
    hashed_value = int(hashlib.sha256(key).hexdigest(), 16)
    mapped_value = (hashed_value % 5000) / 10000.0 + 0.1
    ChaoticRNG=ChoticNumberGenerator(16*block,mapped_value)
    smalleChaoticRNG=split_array(ChaoticRNG, Nov_BLOCK_SIZE)
    #print(ChaoticRNG)
    for j in range(len(plain) // Nov_BLOCK_SIZE):
        p_j = plain[j*Nov_BLOCK_SIZE:(j+1)*Nov_BLOCK_SIZE]
        c_j = Nov_encryption(smalleChaoticRNG[j],p_j, key)
        cipher += c_j
    return bytes(cipher)


def Nov_Blocks_decryption(cipher: bytes, key: bytes) -> bytes:
    plain = []
    block=(len(cipher)+(len(cipher)%8)) // Nov_BLOCK_SIZE
    hashed_value = int(hashlib.sha256(key).hexdigest(), 16)
    mapped_value = (hashed_value % 5000) / 10000.0 + 0.1
    ChaoticRNG=ChoticNumberGenerator(16*block,mapped_value)
    smalleChaoticRNG=split_array(ChaoticRNG, Nov_BLOCK_SIZE)
    
    for j in range(len(cipher) // Nov_BLOCK_SIZE):
        c_j = cipher[j*Nov_BLOCK_SIZE:(j+1)*Nov_BLOCK_SIZE]
        p_j = Nov_decryption(smalleChaoticRNG[j],c_j, key)
        plain += p_j
    return bytes(plain)



def pkcs7_pad(data: bytes, block_size: int) -> bytes:
    pad_length = block_size - len(data) % block_size
    padding = bytes([pad_length] * pad_length)
    return data + padding


def pkcs7_unpad(padded_data: bytes) -> bytes:
    pad_length = padded_data[-1]
    if pad_length > len(padded_data):
        raise ValueError("Invalid padding")
    return padded_data[:-pad_length]

def extract_hex_bytes(bytearray_str):
    byte_str = bytearray_str[len("bytearray(b\""):-2]
    hex_bytes = byte_str.split("\\x")[1:]
    byte_list = bytes.fromhex("".join(hex_bytes))
    return byte_list

def file_to_bytearray(filename):
    try:
        with open(filename, 'rb') as file:
            binary_data = file.read()
            bytearray_str = "bytearray(b\"" + "".join("\\x{:02x}".format(byte) for byte in binary_data) + "\")"
            return bytearray_str
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None

def read_file_to_bytes(filename):
    try:
        with open(filename, 'rb') as file:
            binary_data = file.read()
            return binary_data
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None



def get_file_extension(filename):
    return os.path.splitext(filename)[1]

def store_encrypted_file(ciphertext: bytes, original_filename: str):
    try:
        file_extension = get_file_extension(original_filename)
        encrypted_filename = os.path.splitext(original_filename)[0] + "_encrypted" + file_extension
        with open(encrypted_filename, 'wb') as file:
            file.write(ciphertext)
    except Exception as e:
        print(f"Error storing encrypted data: {e}")


def visualize_encrypted_data(ciphertext, output_filename):
    # Assume ciphertext is a bytes object
    data = np.frombuffer(ciphertext, dtype=np.uint8)
    side_length = int(np.ceil(np.sqrt(len(data))))
    padded_length = side_length ** 2

    if padded_length != len(data):
        data = np.pad(data, (0, padded_length - len(data)), mode='constant', constant_values=0)

    data = data.reshape((side_length, side_length))

    plt.imshow(data, cmap='gray', interpolation='none')
    plt.axis('off')

    # Check the output filename extension
    if output_filename.lower().endswith('.gif'):
        # Change the filename extension to .png for visualization
        output_filename = output_filename.replace('.gif', '.png')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def store_decrypted_file(decrypted_data: bytes, original_filename: str):
    try:
        file_extension = get_file_extension(original_filename)
        decrypted_filename = os.path.splitext(original_filename)[0] + "_decrypted" + file_extension
        with open(decrypted_filename, 'wb') as file:
            file.write(decrypted_data)
    except Exception as e:
        print(f"Error storing decrypted data: {e}")



def generate_Nov_key(user_input):
    user_input_bytes = user_input.encode('utf-8')
    hashed_input = hashlib.sha256(user_input_bytes).digest()
    Nov_key = hashed_input[:16]
    return Nov_key
    

def save_with_metadata(binary_data, original_extension, output_filename):
    data_with_metadata = binary_data + b'\0' + original_extension.encode('utf-8')
    with open(output_filename, 'wb') as file:
        file.write(data_with_metadata)

def read_with_metadata(input_filename):
    with open(input_filename, 'rb') as file:
        data_with_metadata = file.read()
    index_null_byte = data_with_metadata.rfind(b'\0')
    if index_null_byte == -1:
        raise ValueError("Metadata separator not found in the file.")
    binary_data = data_with_metadata[:index_null_byte]
    extracted_extension = data_with_metadata[index_null_byte + 1:].decode('utf-8')
    return binary_data, extracted_extension



def bytes_to_bits(data):
    # Convert bytes to bits
    return ''.join(format(byte, '08b') for byte in data)

def bits_to_bytes(bits):
    # Convert bits to bytes
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

def encrypt(filename, key, BlockSize):
    result = file_to_bytearray(filename)
    Nov_key = generate_Nov_key(key)
    file_name, original_extension = os.path.splitext(filename)
    output_filename = file_name + ".ext"

    if result is not None:
        bytearr = extract_hex_bytes(result)
        padded_data = pkcs7_pad(bytearr, BlockSize)
        ciphertext = Nov_Blocks_encryption(padded_data, Nov_key)
        save_with_metadata(ciphertext, original_extension, output_filename)
        return ciphertext
    else:
        print("Failed to read and encrypt the file data.")

def decrypt(filename, key):
    binary_data_read, extracted_extension = read_with_metadata(filename)
    result = binary_data_read
    Nov_key = generate_Nov_key(key)
    file_name, original_extension = os.path.splitext(filename)
    output_filename_decrypted = file_name + extracted_extension

    if result is not None:
        recovered_plaintext = Nov_Blocks_decryption(binary_data_read, Nov_key)
        original_data = pkcs7_unpad(recovered_plaintext)
        store_decrypted_file(original_data, output_filename_decrypted)
    else:
        print("Failed to read and decrypt the file data.")


BlockSize=32
