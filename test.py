gpstate_raw=[b'0' b'0' b'0' b'23' b'80' b'12' b'12' b'12' b'12' b'12' b'12' b'12'
 b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12'
 b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12'
 b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'12' b'ff' b'12'
 b'12' b'12']

gpstate_int = [int(i.decode('utf-8'), 16) for i in gpstate_raw]
print(gpstate_int)