

def main():
    traj = load('traj.npy')
    x = trajectory.state_pack(traj)
    t = traj['t']
    t,E = construct_basis(t, x)
    E_sp = make_interp_spline(t, E, k=3, bc_type='periodic')

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
