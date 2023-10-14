import numpy as np
from episode import episode_from_ep_name


def names_in_scene(scene):
    return set([line.split(':')[0] for line in scene.strip('\n').split('\n')
                if not line.startswith('[')])

ep = episode_from_ep_name('oltl-10-18-10')
#ep.scenes = ep.scenes[40:]
char_names_in_scenes = [names_in_scene(scene) for scene in ep.scenes]
n = len(ep.scenes)
dists = np.empty([n,n])
for i, cn1 in enumerate(char_names_in_scenes):
    dists[i,i] = 0
    for j in range(i+1,n):
        cn2 = char_names_in_scenes[j]
        n_overlap = sum([x in cn2 for x in cn1])
        iou = n_overlap / (len(cn1)+len(cn2))
        dists[i,j] = 1 - iou
        dists[j,i] = 1 if n_overlap==0 else np.inf # never change order for a char
dists = np.concatenate([dists,np.zeros([n,1])],axis=1) #dummy endpoint

order_idxs = list(range(n+1)) #dummy endpoint
#print(order_idxs)
#print('\n'.join([' '.join(char_names_in_scenes[oi]) for oi in order_idxs[:-1]]))
while True:
    changed = False
    for i in range(2,n):
        prevs_with_matching_chars = [j for j,scene_idx in enumerate(order_idxs[:i]) if dists[scene_idx,order_idxs[i]]<1]
        #if char_names_in_scenes[order_idxs[i]] == set(['Hannah','James']):
            #breakpoint()
        if len(prevs_with_matching_chars) > 0: # move as far to left without inverting char order
            new_pos = max(prevs_with_matching_chars)+1
            if new_pos==i:
                continue
            cost_improvement_at_i = dists[order_idxs[i-1],order_idxs[i]] + \
                                    dists[order_idxs[i],order_idxs[i+1]] - \
                                    dists[order_idxs[i-1],order_idxs[i+1]]

            cost_improvement_at_new_pos = dists[order_idxs[new_pos-1],order_idxs[new_pos]] - \
                                          dists[order_idxs[new_pos-1],order_idxs[i]] - \
                                          dists[order_idxs[i],order_idxs[new_pos]]

            if cost_improvement_at_i + cost_improvement_at_new_pos <= 0:
                continue
            old_costs = [dists[order_idxs[k],order_idxs[k+1]] for k in range(n-1)]
            #print(order_idxs)
            #print(old_costs)
            #print(f'moving {i}--{char_names_in_scenes[order_idxs[i]]} to {new_pos}')
            to_move = order_idxs.pop(i)
            order_idxs.insert(new_pos,to_move)
            changed = True
            new_costs = [dists[order_idxs[k],order_idxs[k+1]] for k in range(n-1)]
            #print(order_idxs)
            #print(new_costs)
            #print(sum(old_costs),sum(new_costs))
            if not np.allclose(sum(old_costs)-sum(new_costs), cost_improvement_at_i+cost_improvement_at_new_pos):
                breakpoint()
    if not changed:
        break
    print(order_idxs)
    print('\n'.join([' '.join(char_names_in_scenes[oi]) for oi in order_idxs[:-1]]))

"""
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    if best_improvement > 0:
        #print(best_improvement)
        print(move)
        i,j,k = move
        idxs_wo_chunk = order_idxs[:i] + order_idxs[j:]
        order_idxs = idxs_wo_chunk[:k] + order_idxs[i:j] + idxs_wo_chunk[k:]
        print(order_idxs)
        assert set(order_idxs) == set(range(n+1))
        assert len(order_idxs) == n+1
        assert order_idxs[0] == 0
        assert order_idxs[n] == n
    else:
        break
print(order_idxs)
new_cns = [ep.scenes[i] for i in order_idxs]
print(new_cns)

"""

