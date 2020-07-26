#!/usr/bin/env python
import sys

def il(n):
  return n+1

def ol_den(n):
  return n

# for decode
def ol_dec(n):
  return n+1

def s(n):
  return n


if __name__ == "__main__":
  mode = sys.argv[1]
  with open(sys.argv[2]) as f:
    lines = f.readlines()
  phone_count = 0
  disambig_count = 0
  for line in lines:
    sp = line.split()
    phone = sp[0]
    if phone == '<eps>' or phone == '<blk>':
      continue
    if phone.startswith('#'):
      disambig_count += 1
    else:
      phone_count += 1

  if mode == "den":
    ol = ol_den
  elif mode == "decode":
    ol = ol_dec
  else:
    print("mode error!")
    exit(-1)

  print('0 0 {} 0'.format(il(0)))
  for i in range(1, phone_count+1):
    print('0 {} {} {}'.format(s(i), il(i), ol(i)))
    print('{} {} {} 0'.format(s(i), s(i), il(i)))
    print('{} 0 {} 0'.format(s(i), il(0)))
  
  for i in range(1, phone_count+1):
    for j in range(1, phone_count+1):
      if i != j:
        print('{} {} {} {}'.format(s(i), s(j), il(j), ol(j)))

  for i in range(0, phone_count+1):
    print(s(i))
    if mode == "decode":
      for j in range(phone_count+2, phone_count+disambig_count+2):
        print('{} {} {} {}'.format(s(i), s(i), 0, j))
