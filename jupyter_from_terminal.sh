function runnb {
    jupyter nbconvert --to script --output "temp_script"  "$1"  #"${BASH_SOURCE[1]}";
    python3 temp_script.py "${@:2}";
}

