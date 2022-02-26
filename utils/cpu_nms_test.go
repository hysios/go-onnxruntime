package utils

import (
	"testing"
)

var boxes = [][]float32{
	[]float32{564.3192, 68.852196, 596.58685, 92.542885, 0.75834227, 0.0008434057, 0.0022590458, 0.0009969771, 0.9940401, 0.001157105, 0.0009993911, 0.0010488331, 0.0056681335, 0.0008662641, 0.00079238415},
	[]float32{249.69911, 72.09886, 283.05154, 95.16132, 0.71412486, 0.0013934672, 0.0035797954, 0.0013181269, 0.98670495, 0.0019522905, 0.0016245246, 0.0009469986, 0.010220677, 0.0011155903, 0.0008402467},
	[]float32{250.90668, 72.363815, 282.40155, 96.0392, 0.7989, 0.0014843345, 0.00230062, 0.0012622774, 0.98665214, 0.0019577742, 0.0016986132, 0.0012753904, 0.011435926, 0.001459986, 0.00086960196},
	[]float32{254.32999, 73.20302, 283.39114, 96.952614, 0.46746126, 0.0022639334, 0.0028916895, 0.0010296702, 0.98945487, 0.00216043, 0.0016082525, 0.0011322796, 0.005963117, 0.0011147261, 0.00084903836},
	[]float32{564.8234, 68.97817, 595.9336, 92.34203, 0.7264527, 0.001852721, 0.0022262037, 0.0013275743, 0.9896592, 0.0015488267, 0.0014965534, 0.0011884272, 0.008079439, 0.0013792515, 0.000939101},
	[]float32{566.6109, 69.17494, 596.0929, 92.29683, 0.29455137, 0.0034668148, 0.0035665631, 0.0012654066, 0.98215556, 0.0021267831, 0.0017923415, 0.0012668967, 0.006372303, 0.0014213622, 0.0010639131},
	[]float32{252.01425, 73.13071, 282.793, 96.54076, 0.5018133, 0.0037462711, 0.004452288, 0.0014275014, 0.98336697, 0.0022710562, 0.0017644763, 0.0012138784, 0.008708268, 0.0010707676, 0.0008662045},
	[]float32{557.1304, 203.40392, 578.781, 209.06477, 0.35617554, 0.9990085, 0.0013668239, 0.000646472, 0.002850175, 0.0008830428, 0.001044184, 0.0008122623, 0.0007144809, 0.0015438795, 0.002185315},
	[]float32{557.28424, 203.29872, 578.09955, 209.10081, 0.32164717, 0.99914974, 0.00089359283, 0.000865221, 0.0017353892, 0.0007457733, 0.00054079294, 0.0006072223, 0.0006598532, 0.0010418892, 0.003955573},
	[]float32{564.3859, 68.86633, 596.51434, 92.5636, 0.77141345, 0.000510633, 0.0015862882, 0.0006200969, 0.99549925, 0.00073748827, 0.00067543983, 0.00064477324, 0.004479319, 0.0006753504, 0.00066176057},
	[]float32{249.77635, 72.143074, 283.17142, 94.90236, 0.7025708, 0.00082582235, 0.0022129118, 0.0009345412, 0.9904118, 0.0013862848, 0.001073271, 0.00066420436, 0.009036183, 0.0007581413, 0.0005303323},
	[]float32{250.94983, 72.281586, 282.73053, 95.922714, 0.8070409, 0.000736326, 0.0019492805, 0.0009901524, 0.9895448, 0.001562506, 0.001180768, 0.00090417266, 0.01019612, 0.0011900067, 0.00055897236},
	[]float32{254.21776, 73.38076, 283.54916, 96.76937, 0.49907434, 0.0014611185, 0.002644956, 0.0006790757, 0.99251175, 0.0012183189, 0.0010440946, 0.000670135, 0.0048612654, 0.0006825924, 0.00060388446},
	[]float32{557.85785, 68.77485, 596.9654, 92.40822, 0.38075435, 0.001851201, 0.0033141673, 0.001219064, 0.9865867, 0.0019182861, 0.0013945401, 0.0008665323, 0.006717384, 0.0009137094, 0.0007888675},
	[]float32{565.0746, 69.05441, 596.1997, 92.15388, 0.7162249, 0.00082954764, 0.0017573237, 0.0009288788, 0.99120575, 0.0014727414, 0.0011109412, 0.00088617206, 0.0070010126, 0.0009765923, 0.00059226155},
	[]float32{566.38904, 69.15511, 596.4215, 92.167725, 0.3096314, 0.0020368993, 0.00285694, 0.00083115697, 0.98771024, 0.0013250113, 0.0012943149, 0.00084781647, 0.004830897, 0.0007645786, 0.00081041455},
	[]float32{96.05136, 66.80871, 153.60402, 131.22083, 0.40883064, 0.00079870224, 0.044305444, 0.0006069243, 0.35253698, 0.0027950406, 0.08560586, 0.0024403632, 0.35600603, 0.00041237473, 0.0006161332},
	[]float32{252.07953, 73.08596, 282.9525, 96.5077, 0.48954627, 0.0017585754, 0.0032607317, 0.0010733306, 0.9872054, 0.0013830662, 0.0011612475, 0.00080037117, 0.0076806545, 0.0009908676, 0.0005929172},
	[]float32{95.02818, 69.0294, 150.96857, 131.1539, 0.56465966, 0.0018983781, 0.0018048286, 0.0014309287, 0.33664018, 0.0014486015, 0.2537883, 0.004185915, 0.26296574, 0.00062960386, 0.0023726523},
	[]float32{95.29024, 67.92256, 152.71286, 131.40121, 0.9220083, 0.00055485964, 0.0022406578, 0.0013439059, 0.36272395, 0.0009454787, 0.26245624, 0.0034164786, 0.2221218, 0.0004491806, 0.0015613139},
	[]float32{97.08461, 67.737, 153.58356, 132.11554, 0.8101769, 0.0012010932, 0.001995355, 0.0010058582, 0.4130776, 0.0011177957, 0.17166921, 0.0029025972, 0.28768152, 0.00035342574, 0.001777947},
	[]float32{416.91342, 68.39412, 474.13614, 135.47592, 0.9058381, 0.00037404895, 0.0053755343, 0.0012099743, 0.15023243, 0.0009276867, 0.45481557, 0.003281206, 0.2860456, 0.00032272935, 0.0015862882},
	[]float32{416.81198, 67.74815, 474.92166, 136.46266, 0.7374459, 0.00024151802, 0.039416164, 0.0007407665, 0.07095903, 0.00073072314, 0.47870097, 0.0023552477, 0.26454192, 0.00017747283, 0.0011163652},
	[]float32{95.51466, 69.237564, 152.24597, 132.45663, 0.43348104, 0.0020224154, 0.0032274127, 0.0016674697, 0.483309, 0.002027601, 0.16098124, 0.0040026307, 0.17266604, 0.0008081198, 0.001912415},
	[]float32{416.85394, 68.81631, 473.85364, 136.09845, 0.8443247, 0.0007363856, 0.0026943386, 0.0017047524, 0.2654082, 0.0013675094, 0.42943382, 0.0045233965, 0.324498, 0.0006676614, 0.0020101368},
	[]float32{564.23944, 68.86231, 596.49164, 92.57534, 0.77411413, 0.00051248074, 0.0011272728, 0.00051921606, 0.9966018, 0.0005787611, 0.00067701936, 0.000372231, 0.0037221014, 0.0005764365, 0.00045356154},
	[]float32{249.83017, 72.171715, 283.01175, 94.89824, 0.70249707, 0.0005951524, 0.0015800297, 0.00081029534, 0.9920571, 0.0011212826, 0.0010695755, 0.00047427416, 0.008566827, 0.0008198619, 0.0005185306},
	[]float32{250.86139, 72.29158, 282.7315, 95.902954, 0.806536, 0.0007054508, 0.001665771, 0.00081446767, 0.99053425, 0.0011070371, 0.0012705326, 0.0005891025, 0.009560972, 0.0011267662, 0.0005888641},
	[]float32{254.22246, 73.380615, 283.5565, 96.74658, 0.49061015, 0.0011172295, 0.0017058849, 0.00070667267, 0.99267447, 0.0012052655, 0.00096154213, 0.00044700503, 0.0043076277, 0.00068330765, 0.0006625056},
	[]float32{557.82495, 68.767715, 596.6861, 92.43535, 0.39595228, 0.0014370382, 0.0027965307, 0.0010211766, 0.98856956, 0.0014599562, 0.0012557507, 0.0008355379, 0.006077498, 0.0008558631, 0.00078460574},
	[]float32{564.73035, 69.04366, 596.24744, 92.14928, 0.71849275, 0.0007148683, 0.0014587343, 0.000687778, 0.9928398, 0.0010075569, 0.0010822117, 0.0005533993, 0.0062926114, 0.00095260143, 0.0005763769},
	[]float32{566.1917, 69.136185, 596.6219, 92.19233, 0.31128025, 0.0015176237, 0.002159059, 0.0007415712, 0.9886613, 0.0010668933, 0.0011505187, 0.00058194995, 0.004574269, 0.00070762634, 0.0007112324},
	[]float32{96.43682, 65.982376, 153.71524, 132.40717, 0.4438808, 0.0012453198, 0.029369146, 0.0006314814, 0.3612788, 0.0025346577, 0.100117266, 0.002292037, 0.41408473, 0.00014969707, 0.00027546287},
	[]float32{252.05624, 73.00413, 282.88492, 96.566345, 0.4946935, 0.0018586814, 0.00260669, 0.00097322464, 0.98836815, 0.0012427568, 0.0011833608, 0.0005450845, 0.0072251856, 0.00084781647, 0.0006558299},
	[]float32{95.639114, 68.7731, 150.29903, 131.24435, 0.58311874, 0.0016648471, 0.0022514462, 0.0013183057, 0.3104096, 0.0015211403, 0.24618003, 0.004104942, 0.26650512, 0.0002591908, 0.0015799105},
	[]float32{95.962425, 67.61342, 152.32304, 131.57478, 0.933334, 0.00057554245, 0.0014813244, 0.0012434423, 0.38561213, 0.00069886446, 0.23825392, 0.0026368499, 0.23938337, 0.00023561716, 0.00090277195},
	[]float32{97.583115, 67.08981, 153.10977, 132.60379, 0.82424355, 0.0012464523, 0.0015020072, 0.00086277723, 0.42802447, 0.0011083484, 0.18238708, 0.002695918, 0.30312365, 0.00019049644, 0.0010577738},
	[]float32{417.28555, 67.75271, 473.89587, 136.04031, 0.9240111, 0.00037139654, 0.0037799776, 0.00106439, 0.17606512, 0.00076943636, 0.44470966, 0.0031514466, 0.31443763, 0.00016358495, 0.0009687245},
	[]float32{416.66275, 66.928055, 474.98108, 137.32513, 0.76345766, 0.0002991557, 0.025785, 0.000672698, 0.08431828, 0.00075376034, 0.5268929, 0.0028821826, 0.28694323, 7.900596e-05, 0.0006233156},
	[]float32{95.56457, 68.61533, 152.0748, 133.00623, 0.45432097, 0.0038226545, 0.0030955374, 0.001463443, 0.40847132, 0.0019500554, 0.17021644, 0.0034377873, 0.18341893, 0.00038149953, 0.0012254119},
	[]float32{417.21622, 68.28136, 473.54352, 136.50476, 0.85958767, 0.0009740591, 0.0023435354, 0.0015020967, 0.24982461, 0.0013443232, 0.43351418, 0.004426867, 0.3471753, 0.00030225515, 0.0011160374},
	[]float32{196.7305, 145.6956, 264.24188, 199.20006, 0.9085176, 0.0014659166, 0.0008228421, 0.00095435977, 0.974692, 0.001463592, 0.0015051663, 0.00080895424, 0.02238375, 0.0004041195, 0.00036633015},
	[]float32{196.79639, 145.97949, 264.23572, 199.33649, 0.8549241, 0.0037513077, 0.0011662841, 0.0009788573, 0.94455457, 0.003212154, 0.0015648901, 0.0010784268, 0.018818647, 0.00042942166, 0.00047135353},
	[]float32{198.13985, 146.46364, 264.48386, 199.81006, 0.34724027, 0.0021030307, 0.0010082424, 0.0010611713, 0.9864472, 0.0018568039, 0.0021309555, 0.00084519386, 0.012941241, 0.00041514635, 0.00038775802},
	[]float32{523.6418, 156.01219, 596.8996, 217.61888, 0.36191672, 0.004608929, 0.016756237, 0.0008315444, 0.72625804, 0.0036233962, 0.005709201, 0.0018861592, 0.041717082, 0.000230968, 0.00025996566},
	[]float32{525.5222, 158.43416, 597.7073, 217.24979, 0.8964927, 0.0008827746, 0.0006734133, 0.00068345666, 0.95455635, 0.0017906725, 0.004206866, 0.0009814799, 0.060611248, 0.00027391315, 0.00027266145},
	[]float32{524.6326, 158.7047, 596.9911, 216.94447, 0.8990065, 0.0015908182, 0.0004530549, 0.0007792413, 0.9613053, 0.0010602772, 0.0032179356, 0.00085034966, 0.017913759, 0.00027042627, 0.0003157258},
	[]float32{308.69916, 477.67267, 416.67566, 505.0789, 0.4117887, 0.9964968, 0.00029924512, 0.00019988418, 0.002083093, 0.0003335476, 0.00025823712, 0.0004170239, 0.001450777, 0.00020125508, 0.00041505694},
	[]float32{307.71265, 478.91913, 416.6168, 506.2187, 0.81400937, 0.9957142, 0.0003296733, 0.0003131628, 0.0014974177, 0.0003605783, 0.0001834631, 0.00082471967, 0.00066778064, 0.00052911043, 0.00056278706},
	[]float32{308.73538, 478.54233, 415.41895, 505.83176, 0.8838875, 0.9888394, 0.0002310574, 0.00051659346, 0.0033803582, 0.0008585453, 0.00028303266, 0.0009433925, 0.0022434592, 0.00086060166, 0.00041371584},
	[]float32{308.99997, 479.4248, 416.22314, 506.05582, 0.6873963, 0.94578815, 0.0004901588, 0.00090640783, 0.008185297, 0.0014359057, 0.0007470548, 0.0012865961, 0.0043675303, 0.00075683, 0.0011156201},
	[]float32{249.94127, 72.300354, 283.22803, 96.01658, 0.8033857, 0.002020508, 0.0037401617, 0.0011132658, 0.9859532, 0.0013171732, 0.0018078685, 0.00095674396, 0.009919286, 0.0009821951, 0.0009533167},
	[]float32{564.17535, 68.77985, 596.4236, 92.44787, 0.73028195, 0.0015412569, 0.003615886, 0.0012604594, 0.9906297, 0.0014481246, 0.0018701553, 0.0010870695, 0.0069940984, 0.001157552, 0.0010615885},
	[]float32{96.92157, 67.47333, 152.52556, 131.12573, 0.9062743, 0.0011247694, 0.0038401484, 0.0017598867, 0.30461264, 0.0016546249, 0.28114378, 0.0021685362, 0.32639182, 0.0016386807, 0.00121665},
	[]float32{250.48624, 72.304184, 283.4623, 96.075325, 0.804628, 0.0019550025, 0.0033954084, 0.0014103055, 0.98899853, 0.0017178953, 0.0014411807, 0.0010814369, 0.008094221, 0.0013071299, 0.0012104511},
	[]float32{251.55342, 72.50255, 283.00873, 96.15584, 0.8280548, 0.0022946894, 0.003683269, 0.001509428, 0.98854125, 0.0017722845, 0.0015276968, 0.0012470484, 0.0067508817, 0.0013872981, 0.0015523136},
	[]float32{416.6177, 67.77915, 474.30804, 136.61533, 0.8344471, 0.00071805716, 0.005405396, 0.0013604462, 0.11446038, 0.0014791787, 0.41804057, 0.0023671389, 0.3953496, 0.0014537275, 0.0009799898},
	[]float32{562.5, 68.69761, 596.1627, 92.40041, 0.623642, 0.002730757, 0.004670471, 0.0020213425, 0.97901416, 0.0016259849, 0.00185287, 0.0011148751, 0.014285386, 0.0012280643, 0.0013222694},
	[]float32{564.186, 68.55052, 596.33057, 92.11029, 0.7174115, 0.0018402636, 0.0030947626, 0.0014289618, 0.9918497, 0.0016914904, 0.001257807, 0.0011623502, 0.005885184, 0.0012486875, 0.0011957288},
	[]float32{96.77394, 68.44636, 152.66525, 131.16644, 0.9256337, 0.0019187629, 0.0036946237, 0.002254188, 0.4149853, 0.002229631, 0.2660799, 0.0032062829, 0.27100235, 0.001987189, 0.0015503168},
	[]float32{97.01415, 67.713394, 152.81664, 131.6075, 0.93271554, 0.0013601482, 0.0034537315, 0.0012533665, 0.27681828, 0.0015627742, 0.30968755, 0.0025915802, 0.31705308, 0.0014855266, 0.0014181137},
	[]float32{416.7803, 68.46686, 473.80676, 135.65973, 0.91482943, 0.0015430748, 0.0048246086, 0.0018586516, 0.16050068, 0.0019412637, 0.5174839, 0.0034261346, 0.34332448, 0.0020009875, 0.0015771687},
	[]float32{417.1306, 68.02565, 473.64407, 135.80014, 0.91969264, 0.0014310777, 0.004787445, 0.001416862, 0.13378263, 0.0017345846, 0.4784074, 0.0032106936, 0.37241367, 0.0018044412, 0.0013556778},
	[]float32{197.91422, 145.24385, 263.34882, 199.93121, 0.7866971, 0.0014054179, 0.007746935, 0.0017960072, 0.94283354, 0.0026114583, 0.0026908517, 0.0016161203, 0.019887716, 0.0019378662, 0.0016079545},
	[]float32{198.0191, 145.60104, 264.88147, 199.69447, 0.9310891, 0.002149552, 0.004261732, 0.0021821558, 0.9716505, 0.003578633, 0.003014952, 0.0022161305, 0.019034266, 0.0025436878, 0.0021695495},
	[]float32{198.61311, 145.6794, 264.4516, 200.01166, 0.9237032, 0.0019436777, 0.0030149817, 0.0019893944, 0.9792606, 0.0028010309, 0.0025625527, 0.0020189881, 0.0148651, 0.0017473698, 0.001589328},
	[]float32{525.0627, 157.51031, 597.1373, 217.5351, 0.9201771, 0.0021217465, 0.0035670102, 0.0020347238, 0.95054436, 0.0032131374, 0.0037846267, 0.002225697, 0.03637138, 0.002532959, 0.0020818412},
	[]float32{525.13916, 157.52704, 597.33, 217.07063, 0.92549145, 0.002498567, 0.00426054, 0.0020166636, 0.9580754, 0.0038175583, 0.0041872263, 0.0024264157, 0.028147757, 0.0028226674, 0.0021653473},
	[]float32{526.05, 158.26758, 597.73486, 216.7917, 0.79948586, 0.0018592775, 0.003121525, 0.0019547641, 0.97363293, 0.0030519366, 0.0038288832, 0.002269715, 0.019172758, 0.0017621517, 0.0013797283},
	[]float32{308.32474, 478.7841, 415.5637, 506.15878, 0.89782155, 0.99688923, 0.0006414652, 0.00072202086, 0.0011838377, 0.0016534925, 0.0009290576, 0.0013062358, 0.0010891855, 0.0014001429, 0.0008148551},
	[]float32{310.05807, 478.23358, 414.93723, 505.93118, 0.886297, 0.996273, 0.0005861819, 0.0009010732, 0.0009893775, 0.0010870993, 0.001038909, 0.0008018911, 0.0012289584, 0.0011712015, 0.0007895231},
	[]float32{309.24512, 479.11435, 415.7553, 506.25165, 0.89637566, 0.99691, 0.00070148706, 0.0007454455, 0.0012224615, 0.0022136867, 0.00082314014, 0.0012173653, 0.0011184216, 0.002151817, 0.00075623393},
	[]float32{249.92813, 72.28143, 283.1295, 95.91241, 0.79688334, 0.0019100308, 0.0012503564, 0.0012216568, 0.98922443, 0.001296103, 0.0017777383, 0.000990808, 0.009126067, 0.0010770261, 0.001182884},
	[]float32{564.2634, 68.75933, 596.3967, 92.37425, 0.7389424, 0.0017443001, 0.0012430251, 0.001465112, 0.99304605, 0.0014395118, 0.0017721653, 0.0010561943, 0.006346911, 0.0011649728, 0.0012479126},
	[]float32{96.891716, 67.45604, 152.54407, 131.19423, 0.9001055, 0.0015853941, 0.0039201677, 0.0023057163, 0.29633945, 0.0014050305, 0.29960012, 0.0022351444, 0.3023228, 0.0014278591, 0.0011989772},
	[]float32{250.47375, 72.24626, 283.4834, 96.014915, 0.8211263, 0.001819849, 0.0013619065, 0.0014061928, 0.9914792, 0.0014996231, 0.0013620555, 0.0010781288, 0.007994413, 0.0012919009, 0.0012142956},
	[]float32{251.57077, 72.52203, 283.08115, 96.18827, 0.8144376, 0.0025222003, 0.0014791191, 0.0015870631, 0.99022084, 0.0015627742, 0.001521647, 0.0009846091, 0.0064943135, 0.0014509261, 0.0013202131},
	[]float32{416.86334, 67.763504, 474.0046, 136.78473, 0.82741416, 0.0011403561, 0.007982612, 0.00214234, 0.09893942, 0.0014537871, 0.45363423, 0.0021439195, 0.35645628, 0.0014875829, 0.0010011494},
	[]float32{562.6351, 68.65465, 596.05365, 92.34009, 0.60844004, 0.0026541054, 0.0018082261, 0.001853615, 0.983405, 0.0014619827, 0.0018103421, 0.0010968149, 0.014099568, 0.0011741519, 0.0013174117},
	[]float32{564.0762, 68.491615, 596.35077, 92.1771, 0.7333302, 0.0016801953, 0.0012551248, 0.001336515, 0.9932594, 0.0014640391, 0.0012500882, 0.0010183454, 0.005659044, 0.001323849, 0.0011718571},
	[]float32{96.9236, 68.64267, 152.87848, 131.05405, 0.922241, 0.002853632, 0.0037684739, 0.0023330152, 0.41940433, 0.0017602146, 0.269591, 0.0029886067, 0.25587675, 0.001496911, 0.0014705956},
	[]float32{97.2344, 67.79717, 152.62044, 131.75252, 0.9309832, 0.0015468597, 0.0029384792, 0.0013271868, 0.29281533, 0.0011559427, 0.32430497, 0.001861006, 0.31906486, 0.0011298954, 0.0012381673},
	[]float32{417.10498, 68.492134, 473.80356, 135.6337, 0.91417867, 0.0021155477, 0.0053827465, 0.002115637, 0.15569392, 0.0015751421, 0.53289944, 0.0032897592, 0.31483412, 0.001463145, 0.0015003681},
	[]float32{417.27945, 68.01777, 473.4901, 135.98853, 0.92115057, 0.0015127361, 0.0042589903, 0.0015378594, 0.14141491, 0.0013139546, 0.49779266, 0.0024494827, 0.3740421, 0.0013593137, 0.0012265444},
	[]float32{197.75739, 145.48419, 263.49786, 199.94566, 0.7924172, 0.0027247071, 0.007297039, 0.0018674433, 0.9489317, 0.0022675097, 0.002692461, 0.00145486, 0.01923871, 0.0019997954, 0.0017476976},
	[]float32{198.15303, 145.63489, 264.8108, 199.65797, 0.9294863, 0.002548337, 0.0039303303, 0.0022776425, 0.9736697, 0.0028588176, 0.002743721, 0.002380252, 0.017953187, 0.0023309588, 0.0023718774},
	[]float32{198.77534, 145.69676, 264.27103, 200.00589, 0.9217619, 0.0025747418, 0.0028538704, 0.0017396808, 0.98229706, 0.0022886395, 0.002149582, 0.0016228855, 0.01403904, 0.0018772185, 0.0018282533},
	[]float32{525.2086, 157.57051, 597.12634, 217.57166, 0.91930556, 0.0031697154, 0.004109919, 0.0020890236, 0.95080256, 0.0026919246, 0.0034198463, 0.0021374226, 0.03388813, 0.0022055805, 0.0023188293},
	[]float32{525.19745, 157.52585, 597.31903, 217.098, 0.923751, 0.003428638, 0.004034281, 0.0020644069, 0.9582521, 0.0030025542, 0.0037488043, 0.0025232434, 0.026409417, 0.002559334, 0.0025134683},
	[]float32{526.1039, 158.23895, 597.71967, 216.75212, 0.78573847, 0.0031948984, 0.002924621, 0.0016647279, 0.97323895, 0.002683252, 0.0033302307, 0.001932174, 0.01780963, 0.0018535852, 0.0018194318},
	[]float32{307.73816, 478.8138, 416.26004, 506.18283, 0.8982084, 0.9968076, 0.0004647076, 0.00060242414, 0.0006251335, 0.0012038052, 0.0009331703, 0.0008966923, 0.0007599592, 0.00076678395, 0.0006119907},
	[]float32{308.3527, 478.32565, 416.6874, 505.73325, 0.86885536, 0.99774873, 0.00041660666, 0.00066944957, 0.00052574277, 0.0011095703, 0.0010477304, 0.00066304207, 0.00081446767, 0.00080770254, 0.00035616755},
	[]float32{308.5279, 479.07605, 416.39404, 506.44073, 0.8901406, 0.9973138, 0.00040772557, 0.0005567968, 0.00061783195, 0.0014878213, 0.0007379651, 0.00092473626, 0.00084125996, 0.0012051165, 0.00049450994},
	[]float32{96.856064, 67.068245, 152.49677, 131.46854, 0.89894885, 0.0015519559, 0.001459986, 0.0021340847, 0.31277522, 0.0012055635, 0.2927431, 0.001886636, 0.28815222, 0.0013850629, 0.0012276769},
	[]float32{416.58258, 67.43536, 474.22086, 136.82703, 0.81742245, 0.0012306571, 0.001113832, 0.0017106235, 0.11338535, 0.0011069775, 0.4455458, 0.0016399622, 0.3644843, 0.0011050999, 0.00090003014},
	[]float32{96.80574, 68.02052, 152.75198, 131.49055, 0.9160545, 0.0017827451, 0.0021226406, 0.0023945272, 0.41946346, 0.0017670691, 0.26669276, 0.003008306, 0.23774934, 0.0016012192, 0.0013928413},
	[]float32{97.078415, 67.6317, 152.6412, 131.56656, 0.9224292, 0.0014276206, 0.0017675161, 0.0013473928, 0.2859461, 0.0012412667, 0.32206237, 0.0022720098, 0.2988885, 0.001303941, 0.0012703836},
	[]float32{416.81857, 67.99856, 473.76053, 135.65085, 0.90788054, 0.0016266108, 0.0020331442, 0.00210163, 0.14641675, 0.0014133155, 0.5349961, 0.003222227, 0.30521968, 0.0014765859, 0.0013269186},
	[]float32{417.11633, 67.859406, 473.53824, 135.50293, 0.9083544, 0.0013765097, 0.002023369, 0.0015261173, 0.1340844, 0.0012812614, 0.49610803, 0.0027905703, 0.35207152, 0.0015022755, 0.0011413395},
	[]float32{197.89153, 145.4929, 263.30658, 199.80928, 0.78148127, 0.002187103, 0.004083574, 0.0022674203, 0.95961976, 0.0025319457, 0.002865851, 0.0016077161, 0.017787963, 0.002162397, 0.0021954477},
	[]float32{198.3198, 145.45926, 264.75272, 199.69452, 0.92042315, 0.0025025308, 0.0033847988, 0.0027222037, 0.9769695, 0.0034328997, 0.0028281212, 0.0023477674, 0.016482204, 0.002596587, 0.0025425553},
	[]float32{198.72292, 145.21852, 264.3479, 200.35454, 0.90753984, 0.0019110441, 0.003022164, 0.0020439923, 0.9829667, 0.002917111, 0.002432406, 0.0017991364, 0.014945686, 0.0018900931, 0.0018232763},
	[]float32{525.2671, 157.71414, 597.03015, 217.17503, 0.9082248, 0.0025528967, 0.002831161, 0.0028149188, 0.9580504, 0.003149718, 0.0038045645, 0.0024639368, 0.03185153, 0.0021602213, 0.002325505},
	[]float32{525.3467, 157.45253, 597.412, 217.00983, 0.91064596, 0.0025521815, 0.0032405853, 0.0028371215, 0.9650524, 0.0036247969, 0.0039053857, 0.002706945, 0.024881244, 0.0026764274, 0.0023615062},
	[]float32{526.10046, 158.1957, 597.8117, 216.75195, 0.77795094, 0.0019264817, 0.002759844, 0.0019642413, 0.97656673, 0.0029532015, 0.0036264956, 0.0018248558, 0.019667, 0.0018019974, 0.0014641881},
	[]float32{95.80236, 68.46207, 152.84369, 132.05774, 0.9013438, 0.0018572807, 0.002958268, 0.0019699633, 0.3530737, 0.0016252697, 0.31558347, 0.0028933585, 0.28148466, 0.0019863248, 0.0023458302},
	[]float32{250.38417, 70.80708, 283.4595, 96.178604, 0.52463585, 0.0030932724, 0.0025688112, 0.0019187629, 0.98582447, 0.0013209581, 0.004084736, 0.0024207532, 0.011129588, 0.0023892224, 0.0031672418},
	[]float32{250.6577, 71.76086, 282.75, 96.552795, 0.5098347, 0.0024450123, 0.002382785, 0.0021445155, 0.98566425, 0.0019617677, 0.0033078194, 0.0021312833, 0.013112396, 0.0024206042, 0.0021634102},
	[]float32{416.15604, 68.651726, 474.39136, 136.75116, 0.8888836, 0.0018441379, 0.0032448173, 0.002147019, 0.156867, 0.0015766621, 0.47395942, 0.0039762855, 0.29288173, 0.0020716786, 0.0024689436},
	[]float32{561.49524, 67.87784, 596.6491, 93.18583, 0.5421828, 0.0023296773, 0.0023608208, 0.0018619299, 0.98922986, 0.0013090372, 0.003533721, 0.0024349391, 0.007116854, 0.002308756, 0.0048452616},
	[]float32{563.1004, 68.02016, 596.061, 92.75492, 0.51846474, 0.002246648, 0.0025569499, 0.002080053, 0.989594, 0.0017634034, 0.0036390126, 0.0024992824, 0.008720636, 0.002454847, 0.0027818382},
	[]float32{96.68138, 67.957306, 152.53418, 131.04263, 0.9151617, 0.0018617511, 0.0026374161, 0.0017666817, 0.44595268, 0.0015763342, 0.28938612, 0.00247249, 0.23138586, 0.0017770827, 0.0019494593},
	[]float32{97.183136, 67.95975, 152.70044, 131.06119, 0.91188717, 0.0017724335, 0.0028074384, 0.0015785992, 0.38611946, 0.0015416443, 0.31440917, 0.002264768, 0.2494775, 0.001860857, 0.0019641519},
	[]float32{250.28223, 71.29194, 283.07593, 95.45234, 0.53437036, 0.009977698, 0.002467543, 0.0016027093, 0.9529104, 0.0015205741, 0.0031184554, 0.0018095076, 0.01303944, 0.002162069, 0.0021793544},
	[]float32{417.18408, 67.88113, 473.74942, 136.08755, 0.90175825, 0.0018826723, 0.0027602613, 0.0019032657, 0.22909603, 0.0016362071, 0.48103812, 0.0028966963, 0.25939235, 0.0018596649, 0.0021165311},
	[]float32{417.38284, 68.06923, 473.77524, 136.00302, 0.9058194, 0.0017929971, 0.0027473867, 0.001670599, 0.17871791, 0.001529336, 0.50233567, 0.0025441647, 0.29213303, 0.001904875, 0.002182424},
	[]float32{198.5848, 145.63338, 263.99182, 199.00514, 0.9282068, 0.0015005767, 0.0031953454, 0.0015586913, 0.9875535, 0.0014587045, 0.0033416152, 0.0019010007, 0.008229464, 0.0015289783, 0.0015774369},
	[]float32{198.0243, 145.82457, 263.4282, 199.89694, 0.92383325, 0.001960218, 0.0021358728, 0.0019446313, 0.99018383, 0.0020249784, 0.002536863, 0.0018738508, 0.0055184364, 0.0017988384, 0.0018944442},
	[]float32{198.45476, 145.70699, 264.04663, 199.55252, 0.9301355, 0.002233237, 0.0026434362, 0.0016998351, 0.9867028, 0.0020291805, 0.0030480325, 0.0018828213, 0.008425593, 0.0018367171, 0.0019236803},
	[]float32{524.57465, 157.63873, 596.7549, 216.85484, 0.9195143, 0.0022974908, 0.0026761591, 0.0019834936, 0.97440606, 0.002248913, 0.0061783195, 0.002163589, 0.015929312, 0.0020938516, 0.0021953285},
	[]float32{524.77496, 158.7329, 598.56866, 216.79305, 0.81401926, 0.0015624166, 0.0021039248, 0.0017533004, 0.98532593, 0.0020892918, 0.004315436, 0.0019000471, 0.011847258, 0.0015888214, 0.001891762},
	[]float32{524.90686, 157.95923, 597.1255, 216.96176, 0.91080284, 0.0018589497, 0.002622068, 0.0021881163, 0.98036075, 0.0024527907, 0.0045233965, 0.002365172, 0.016266257, 0.0021066666, 0.0021618605},
	[]float32{146.2968, 242.03741, 521.0046, 547.1962, 0.8945159, 0.0024158359, 0.0015157163, 0.0010354519, 0.6425657, 0.0010126531, 0.2182996, 0.0016191304, 0.13116309, 0.0013407171, 0.0013775826},
	[]float32{136.63559, 239.34853, 519.3419, 549.4229, 0.8863401, 0.0024992824, 0.0013774335, 0.00084376335, 0.5709077, 0.00087994337, 0.23283365, 0.0015846789, 0.14731672, 0.0011584759, 0.0010252595},
	[]float32{141.6501, 242.57002, 522.29846, 547.13513, 0.9517468, 0.0019521713, 0.0012079775, 0.0009389818, 0.66137123, 0.0008338988, 0.1905801, 0.0013208091, 0.111836195, 0.0011458099, 0.0009796321},
	[]float32{309.44107, 478.68414, 415.98428, 505.22083, 0.8703866, 0.99573576, 0.0014933646, 0.0019308627, 0.0031869411, 0.0016610622, 0.00060003996, 0.0013873875, 0.0011503696, 0.0031629205, 0.0013999939},
	[]float32{308.49994, 479.29694, 417.12354, 506.09033, 0.9074039, 0.9940374, 0.001591742, 0.0018084645, 0.0056212842, 0.0013816953, 0.0006662011, 0.0013341308, 0.0011629164, 0.0027465224, 0.0012217462},
	[]float32{310.11182, 479.0458, 415.08173, 505.98056, 0.8914172, 0.9749287, 0.0026058257, 0.0011509061, 0.011977017, 0.0012633502, 0.00092604756, 0.0013616681, 0.0017384589, 0.0021575093, 0.0010668933},
	[]float32{95.96901, 68.588615, 153.07468, 131.92273, 0.9090947, 0.0023781955, 0.0022648275, 0.0022928119, 0.3279997, 0.001853168, 0.2947721, 0.0031193495, 0.28023142, 0.0024460256, 0.0026552975},
	[]float32{416.25525, 69.18278, 474.47485, 136.4216, 0.89542985, 0.0024323761, 0.0024735034, 0.0024789572, 0.14032677, 0.0018330216, 0.43868497, 0.0041083395, 0.28514987, 0.002421856, 0.002675116},
	[]float32{96.90698, 68.20434, 152.75854, 131.07516, 0.91626954, 0.0023558438, 0.002156198, 0.0020470321, 0.39758557, 0.001767695, 0.303202, 0.0029791594, 0.22793326, 0.002059877, 0.0022948086},
	[]float32{96.88115, 68.38282, 152.87955, 131.13167, 0.9118432, 0.0024350584, 0.0019398332, 0.0019092858, 0.3495549, 0.0018377304, 0.3300044, 0.0028443933, 0.25135052, 0.0022199452, 0.00234738},
	[]float32{417.31842, 68.12662, 473.98553, 136.18277, 0.8949218, 0.0024224818, 0.0021488369, 0.0022182167, 0.190891, 0.001840055, 0.49443913, 0.0033642948, 0.25192338, 0.0021383166, 0.0024403632},
	[]float32{417.0699, 68.33778, 473.95282, 136.25267, 0.89233136, 0.002471447, 0.001944989, 0.0021010935, 0.15113091, 0.0018842518, 0.51618665, 0.0031875968, 0.29161996, 0.002234012, 0.0025959313},
	[]float32{198.53862, 145.54942, 264.10165, 199.11238, 0.9119632, 0.0022935867, 0.0034315288, 0.0021542013, 0.98725975, 0.0024341643, 0.004312992, 0.0029349923, 0.009707451, 0.002352506, 0.002490282},
	[]float32{197.94797, 146.10823, 263.506, 199.70238, 0.90477467, 0.0028369725, 0.003168255, 0.0030128956, 0.9890959, 0.0030927956, 0.0039058328, 0.0028210878, 0.0065295696, 0.002802819, 0.0030860305},
	[]float32{198.55328, 145.76491, 264.08588, 199.35983, 0.90944207, 0.003190279, 0.0036233068, 0.0026195347, 0.9850782, 0.0032525957, 0.0049037933, 0.0030092, 0.010189265, 0.0028203428, 0.0029996037},
	[]float32{524.8811, 157.88133, 596.5567, 216.35835, 0.9161303, 0.0030338168, 0.0029522777, 0.0027694702, 0.9722643, 0.0031720996, 0.008759797, 0.002976656, 0.018533796, 0.0030799806, 0.0031213462},
	[]float32{524.8349, 158.56891, 598.3764, 216.49757, 0.8036016, 0.0024595559, 0.0021328032, 0.0021375716, 0.9840224, 0.0027284026, 0.0056940615, 0.0025678575, 0.013198912, 0.0024057329, 0.0023932755},
	[]float32{525.12823, 158.1028, 597.15875, 216.79129, 0.916929, 0.0028862953, 0.0032582283, 0.0026587248, 0.97891355, 0.0028467476, 0.0063130558, 0.0032562912, 0.017614365, 0.002919972, 0.0030758977},
	[]float32{144.55249, 240.90747, 522.5257, 548.73773, 0.88041425, 0.0009893179, 0.0008587539, 0.000977248, 0.6083439, 0.0009394586, 0.2152327, 0.0017093718, 0.12884635, 0.0012323558, 0.0011083186},
	[]float32{135.6329, 241.21605, 521.4608, 547.8519, 0.8747226, 0.00096157193, 0.0008531213, 0.00081703067, 0.52482784, 0.0007702708, 0.24087644, 0.0016798079, 0.1516746, 0.0010078251, 0.0009461343},
	[]float32{139.37978, 242.92236, 523.99945, 547.3612, 0.94072664, 0.00092586875, 0.00090283155, 0.0009138584, 0.64955497, 0.0007966459, 0.1926687, 0.0015284121, 0.11235452, 0.0010005236, 0.0010147393},
	[]float32{142.59874, 240.31148, 522.8649, 549.31824, 0.8460548, 0.0014011264, 0.0012342632, 0.0011900067, 0.52915066, 0.0013132691, 0.23598513, 0.0021370351, 0.13411736, 0.0013464391, 0.001290232},
	[]float32{135.58708, 241.59956, 522.0861, 548.17456, 0.86033475, 0.0011813641, 0.001121521, 0.0009879768, 0.48706612, 0.0010272264, 0.250727, 0.0018172264, 0.15733492, 0.0012574196, 0.0010893643},
	[]float32{138.6019, 243.50673, 523.828, 547.3194, 0.92124724, 0.0013216138, 0.0012093782, 0.0012119412, 0.59154844, 0.0011238754, 0.223535, 0.001833111, 0.12203491, 0.0013105869, 0.0012712479},
}

func TestCpuNMS(t *testing.T) {

	var (
		scores []float32
	)
	for _, box := range boxes {
		// areas = append(areas, (box[2]-box[0])*(box[3]-box[1]))
		scores = append(scores, box[4])
	}

	// var order = argsort(scores)

	filter := CpuNMS(boxes, scores, 0.45)
	t.Logf("filter %v", filter)
}