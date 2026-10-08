"""Keep each simulator/API identity in one serial lane across dataset splits."""
def lanes(groups):
    result={}
    for group in groups:
        team=group[2]
        if team not in (2,4):raise ValueError('Unknown native team/API identity')
        result.setdefault(team,[]).append(group)
    return list(result.values())
