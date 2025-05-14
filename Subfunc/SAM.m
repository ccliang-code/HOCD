function sam = SAM(s1, s2)

sam = real(acos(dot(s1,s2) / norm(s1) / norm(s2)));