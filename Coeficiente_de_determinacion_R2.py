## Librerias
import numpy as np

Carac = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
etiq = np.array([0.169610271922408, 0.283395812542308, 0.386358737510785, 0.470227872390909, 0.433281293764675, 0.600267648212653, 0.738338980436742, 0.790315020494445, 0.877464268422459, 0.84356446225183, 0.96443891694455])
T0=0.0399999999999352
T1=0.15999999999993264
h_x = T0 + T1*Carac

Prome_et=etiq.mean()
Prome_h=h_x.mean()

Numerador=np.zeros(11)
Denomador=np.zeros(11)

for i in range(len(etiq)):
    Numerador[i] = pow(etiq[i]-Prome_et,2)
    Denomador[i] = pow(h_x[i]-Prome_h,2)
    
Numerador=Numerador.sum()/(len(etiq)-1)
Denomador=Denomador.sum()/(len(h_x)-1)
Coeficiente_de_determinacion=Numerador/Denomador