for i in 0 1 2 3 4
do
  python slam.py --config configs/rgbd/tum/fr1_desk.yaml --eval --seed $i
  python slam.py --config configs/rgbd/tum/fr2_xyz.yaml --eval --seed $i
  python slam.py --config configs/rgbd/tum/fr3_office.yaml --eval --seed $i
done