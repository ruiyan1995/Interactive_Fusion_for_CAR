IN_DATA_DIR="dataset/sth_else/videos"
OUT_DATA_DIR="dataset/sth_else/frames"


if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR})
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/%04d.jpg"
  echo $out_name

  ffmpeg -i "${IN_DATA_DIR}/${video}" -r 12 -q:v 3 "${out_name}"
  # exit 1
#   echo "ffmpeg -i ${IN_DATA_DIR}/${video} -r 30 -q:v 1 ${out_name}"
done
