import torch
def all_correlations(x, y):
    N = x.shape[0]

    sigma_x = torch.std(x, dim=0) + 1e-10
    sigma_y = torch.std(y, dim=0) + 1e-10

    x_b = (x - torch.mean(x, dim=0)) / sigma_x
    del x
    del sigma_x
    y_b = (y - torch.mean(y, dim=0)) / sigma_y

    corrs = (x_b.T @ y_b) / N
    
    return corrs
#
##imnet_train = torch.load("activations/ImNet_DinoV2B_-1/train.pt")
##imnet_test = torch.load("activations/ImNet_DinoV2B_-1/test.pt")
#
##A_train = torch.load('conll_train_stack.pt').squeeze() # DeBerta
##A_test = torch.load('conll_test_stack.pt').squeeze()
#
##A_train = torch.load('activations/IMDB_DebertaB_-1/train.pt')
##A_test = torch.load('activations/IMDB_DebertaB_-1/test.pt')
#
#
#A_train = torch.load('activations/WikiArt_ClipVisionB_-1/train.pt')
#A_test = torch.load('activations/WikiArt_ClipVisionB_-1/test.pt')
#
#print('CLIP-WikiArt')
#from extraction import large_diffs, matrysae, topksae, vanillasae, large_diffs_mem
#
#N = 5
#
#print('--- Diffs 6144 ---')
##diffs_list = [
##    large_diffs_mem(A_train)(A_test) for _ in range(N)
##]
#
#diffs_mppc = []
#for i in range(N):
#    print("On average of 10 runs, MPPC = 0.8787534832954407")
#    break
#    for j in range(i+1,N):
#        diffs_mppc.append(
#            all_correlations(
#                large_diffs_mem(A_train)(A_test),
#                large_diffs_mem(A_train)(A_test)
#            ).max(dim=0).values.mean()
#        )
#print(f'On average of {len(diffs_mppc)} runs, MPPC = {torch.tensor(diffs_mppc).mean()}')
##del diffs_list
#
#
#print('--- TopkSAE 6144 ---')
##sae_list = [
##    topksae(A_train)(A_test) for _ in range(N)
##]
#
#sae_mppc = []
#for i in range(N):
#    print("On average of 10 runs, MPPC = 0.8637669682502747")
#    break
#    for j in range(i+1,N):
#        c1 = topksae(A_train)(A_test)
#        c2 = topksae(A_train)(A_test)
#        sae_mppc.append(
#            all_correlations(
#                c1, c2
#            ).max(dim=0).values.mean()
#        )
#        if len(sae_mppc) >= 5: 
#            print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#            raise KeyboardInterrupt()
#        del c1
#        del c2
#print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
##del sae_list
#
#print('--- VanillaSAE 6144 ---')
##sae_list = [
##    vanillasae(A_train)(A_test) for _ in range(N)
##]
#
#sae_mppc = []
#for i in range(N):
#    for j in range(i+1,N):
#        sae_mppc.append(
#            all_correlations(
#                vanillasae(A_train)(A_test),
#                vanillasae(A_train)(A_test)
#            ).max(dim=0).values.mean()
#        )
#        if len(sae_mppc) >= 5: 
#            print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#            break
#            #raise KeyboardInterrupt()
#print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
##del sae_list
#
#print('--- MatryoshkaSAE 6144 ---')
##sae_list = [
##    matrysae(A_train)(A_test) for _ in range(N)
##]
#
#sae_mppc = []
#for i in range(N):
#    for j in range(i+1,N):
#        sae_mppc.append(
#            all_correlations(
#                matrysae(A_train)(A_test),
#                matrysae(A_train)(A_test)
#            ).max(dim=0).values.mean()
#        )
#        if len(sae_mppc) >= 5: 
#            print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#            break
#print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list

#########################################################################################################################################""
#########################################################################################################################################""
#########################################################################################################################################""
#########################################################################################################################################""
#########################################################################################################################################""


A_train = torch.load('activations/IMDB_DebertaB_-1/train.pt')
A_test = torch.load('activations/IMDB_DebertaB_-1/test.pt')

print('Deberta-IMDB')
from extraction import large_diffs, matrysae, topksae, vanillasae, large_diffs_mem

N = 3

print('--- Diffs 6144 ---')
#diffs_list = [
#    large_diffs_mem(A_train)(A_test) for _ in range(N)
#]

diffs_mppc = []
for i in range(N):
    for j in range(i+1,N):
        diffs_mppc.append(
            all_correlations(
                large_diffs_mem(A_train)(A_test),
                large_diffs_mem(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(diffs_mppc)} runs, MPPC = {torch.tensor(diffs_mppc).mean()}')
#del diffs_list


print('--- TopkSAE 6144 ---')
#sae_list = [
#    topksae(A_train)(A_test) for _ in range(N)
#]

sae_mppc = []
for i in range(N):
    for j in range(i+1,N):
        sae_mppc.append(
            all_correlations(
                topksae(A_train)(A_test),
                topksae(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list

print('--- VanillaSAE 6144 ---')
#sae_list = [
#    vanillasae(A_train)(A_test) for _ in range(N)
#]

sae_mppc = []
for i in range(N):
    for j in range(i+1,N):
        sae_mppc.append(
            all_correlations(
                vanillasae(A_train)(A_test),
                vanillasae(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list

print('--- MatryoshkaSAE 6144 ---')
#sae_list = [
#    matrysae(A_train)(A_test) for _ in range(N)
#]

sae_mppc = []
for i in range(N):
    for j in range(i+1,N):
        sae_mppc.append(
            all_correlations(
                matrysae(A_train)(A_test),
                matrysae(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list



##########################################################################################################################################""
##########################################################################################################################################""
##########################################################################################################################################""
##########################################################################################################################################""
##########################################################################################################################################""


A_train = torch.load('activations/AudioSet_ASTB_-1/train.pt')
A_test = torch.load('activations/AudioSet_ASTB_-1/test.pt')

print('AST-Audioset')
from extraction import large_diffs, matrysae, topksae, vanillasae, large_diffs_mem

N = 3

print('--- Diffs 6144 ---')
#diffs_list = [
#    large_diffs_mem(A_train)(A_test) for _ in range(N)
#]

diffs_mppc = []
for i in range(N):
    for j in range(i+1,N):
        diffs_mppc.append(
            all_correlations(
                large_diffs_mem(A_train)(A_test),
                large_diffs_mem(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(diffs_mppc)} runs, MPPC = {torch.tensor(diffs_mppc).mean()}')
#del diffs_list


print('--- TopkSAE 6144 ---')
#sae_list = [
#    topksae(A_train)(A_test) for _ in range(N)
#]

sae_mppc = []
for i in range(N):
    for j in range(i+1,N):
        sae_mppc.append(
            all_correlations(
                topksae(A_train)(A_test),
                topksae(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list

print('--- VanillaSAE 6144 ---')
#sae_list = [
#    vanillasae(A_train)(A_test) for _ in range(N)
#]

sae_mppc = []
for i in range(N):
    for j in range(i+1,N):
        sae_mppc.append(
            all_correlations(
                vanillasae(A_train)(A_test),
                vanillasae(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list

print('--- MatryoshkaSAE 6144 ---')
#sae_list = [
#    matrysae(A_train)(A_test) for _ in range(N)
#]

sae_mppc = []
for i in range(N):
    for j in range(i+1,N):
        sae_mppc.append(
            all_correlations(
                matrysae(A_train)(A_test),
                matrysae(A_train)(A_test)
            ).max(dim=0).values.mean()
        )
print(f'On average of {len(sae_mppc)} runs, MPPC = {torch.tensor(sae_mppc).mean()}')
#del sae_list