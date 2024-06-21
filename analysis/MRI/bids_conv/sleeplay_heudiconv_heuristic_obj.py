import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
	if template is None or not template:
		raise ValueError('Template must be a valid format string')
	return template, outtype, annotation_classes


def infotodict(seqinfo):

    #session = 'ses-01'
    task = 'obj' #'obj' 'sc'
    anat = create_key('sub-{subject}/' + '{session}' + '/anat/sub-{subject}_' + '{session}' + '_T1w')
    task = create_key('sub-{subject}/' + '{session}' + '/func/sub-{subject}_' + '{session}' + '_task-' + task + '_run-{item:02d}_bold')
    fmap_topup = create_key('sub-{subject}/' + '{session}' + '/fmap/sub-{subject}_' + '{session}' + '_dir-{dir}_epi')    

    # anat = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')
    # task = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-obj_run-{item:02d}_bold')
    # fmap_topup = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-{dir}_epi')    


    info = {anat: [], task: [], fmap_topup: []}
    last_run = len(seqinfo)
    for s in seqinfo:
        if 'T1w' in s.series_id: 
            info[anat].append({'item': s.series_id})
        if 'NORM' in s.image_type: 
            if ('dir-AP' in s.series_id): info[fmap_topup].append({'item': s.series_id, 'dir': 'AP'})
            if ('dir-PA' in s.series_id): info[fmap_topup].append({'item': s.series_id, 'dir': 'PA'})	
            if ('run' in s.series_id): info[task].append({'item': s.series_id})
    return info
