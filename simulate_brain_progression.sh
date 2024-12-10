#!/bin/bash

# Use conda brlp environment (activate as necessary)
# conda activate brlp

# Folder containing the CSV files
csv_folder='/home/maia-user/code/InBrainSyn_Revision/BrLP/examples/OASIS3/'

# Files containing CN and AD subject IDs
CN_subs_csv='/home/maia-user/code/InBrainSyn_Revision/BrLP/examples/intersection_CN_subjs.csv'
AD_subs_csv='/home/maia-user/code/InBrainSyn_Revision/BrLP/examples/intersection_AD_subjs.csv'

# Iterate over each CSV file in the folder
for csv_file in "$csv_folder"input.*.d_d*.csv; do
    # Extract the file name (e.g., input.OAS3XXXX.age_XX.csv) and derive SUBID and age
    csv_filename=$(basename "$csv_file")
    SUBID=$(echo "$csv_filename" | sed -E 's/input\.([^.]+)\.d_d[0-9]+\.csv/\1/')

    # Check if SUBID exists in CN_subs_csv or AD_subs_csv
    if grep -q "$SUBID" <(tail -n +2 "$CN_subs_csv" | cut -d ',' -f1); then
        cohort="CN"
        target_diagnosis=1
    elif grep -q "$SUBID" <(tail -n +2 "$AD_subs_csv" | cut -d ',' -f1); then
        cohort="AD"
        target_diagnosis=3
    else
        echo "Skipping $SUBID: Not found in CN or AD lists."
        continue
    fi

    # Output folder for this cohort
    output_folder="/home/maia-user/shared/OASIS3/BrLP.Simul/${cohort}/"

    # Check if the output folder exists, create it if it doesn't
    if [ ! -d "$output_folder" ]; then
        echo "Creating output folder: $output_folder"
        mkdir -p "$output_folder"
    fi

    # Read the `age` from the CSV  <-- New code to load age from CSV
    AGE=$(awk -F, '
        NR == 1 {
            # Find the column index for age
            for (i = 1; i <= NF; i++) {
                if ($i == "age") age_col = i
            }
        }
        NR == 2 {
            # Extract the value in the age column
            print $age_col
        }
    ' "$csv_file" | tr -d '\"' | xargs)

    # Check if `AGE` is valid
    if [ -z "$AGE" ]; then
        echo "Skipping $csv_file: Missing age."
        continue
    fi
    
    # Read the `target_image_uid` from the CSV
    target_image_uid=$(awk -F, '
        NR == 1 {
            # Find the column index for target_image_uid
            for (i = 1; i <= NF; i++) {
                if ($i == "target_image_uid") uid_col = i
            }
        }
        NR == 2 {
            # Extract the value in the target_image_uid column
            print $uid_col
        }
    ' "$csv_file" | tr -d '\"' | xargs)

    # Check if `target_image_uid` is valid
    if [ -z "$target_image_uid" ]; then
        echo "Skipping $csv_file: Missing target_image_uid."
        continue
    fi
    
    # Define the expected output file
    output_file="${output_folder}${SUBID}_${target_image_uid}.nii.gz"

    # Check if the output file already exists
    if [ -f "$output_file" ]; then
        echo "Skipping $csv_file: Output file $output_file already exists."
        continue
    fi
    
    # Debug: Show the extracted values
    echo "DEBUG: Processing SUBID = $SUBID, AGE = $AGE, target_image_uid = $target_image_uid"

    # Construct the command
    cmd="brlp --input $csv_file \
         --output $output_folder \
         --target_age $AGE \
         --customized_name ${SUBID}_${target_image_uid} \
         --target_diagnosis $target_diagnosis \
         --confs ./examples/confs.example.yaml \
         --steps 2"

    # Show the command before running it
    echo "Running command:"
    echo "$cmd"

    # Run the command
    eval "$cmd"
done