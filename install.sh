#!/usr/bin/env bash

# Nebius CLI installation script

set -euo pipefail

VERBOSE=${VERBOSE:-}
if [[ ${VERBOSE} != "" ]]; then
    set -x
fi

# Check if the current os and arch are supported.
UANAME_OS=$(uname -s)
case $UANAME_OS in
    Linux | GNU/Linux)
        OS="linux"
        ;;
    Darwin)
        OS="darwin"
        ;;
    *)
        echo "'$UANAME_OS' os is not supported."
        exit 1
        ;;
esac

UNAME_ARCH=$(uname -m)
case $UNAME_ARCH in
    x86_64 | amd64 | i686-64)
        ARCH="x86_64"
        ;;
    arm64 | aarch64 | aarch64_be | armv8b | armv8l )
        ARCH="arm64"
        ;;
    *)
        echo "'$UNAME_ARCH' architecture is not supported."
        exit 1
        ;;
esac

function curl_with_retry {
    curl -fS --retry 5 --retry-delay 0 --retry-max-time 120 "$@"
}
STORAGE_URL="https://storage.eu-north1.nebius.cloud/cli"

BIN_NAME="nebius"

# Create temporary directory for the downloaded cli.
TMPDIR="${TMPDIR:-/tmp}"
TMP_INSTALL_PATH=$(mktemp -d "${TMPDIR}/${BIN_NAME}-XXXXX")
function cleanup {
    rm -rf "${TMP_INSTALL_PATH}"
}
trap cleanup EXIT

# Check the stable version.
VERSION="${NEBIUS_CLI_VERSION:-$(curl_with_retry -s "${STORAGE_URL}/release/stable")}"
# Download cli to the temporary directory.
echo "Downloading ${BIN_NAME} ${VERSION}"
TMP_CLI="${TMP_INSTALL_PATH}/${BIN_NAME}"
curl_with_retry "${STORAGE_URL}/release/${VERSION}/${OS}/${ARCH}/${BIN_NAME}" -o "${TMP_CLI}"
chmod +x "${TMP_CLI}"

# Check that the cli correctly installed.
set +e
INSTALLED_VERSION=$(${TMP_CLI} version)
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Installation failed. Please contact support. System info: $(uname -a)"
    exit $exit_code
fi
if [ "$VERSION" != "$INSTALLED_VERSION" ]; then
    echo "Installation failed. Expected version is '${VERSION}', but '${INSTALLED_VERSION}' downloaded."
    echo "Please contact support. System info: $(uname -a)"
    exit 1
fi
set -e

CLI_HOME_FOLDER="${HOME}/.nebius"
CUSTOM_INSTALL_FOLDER=${NEBIUS_INSTALL_FOLDER:-}
PATCH_PATH=1

if [[ "${CUSTOM_INSTALL_FOLDER}" != "" ]]; then
    mkdir -p "${CUSTOM_INSTALL_FOLDER}"
    CLI_BIN_FULL_PATH="${CUSTOM_INSTALL_FOLDER%%/}/${BIN_NAME}"
    PATCH_PATH=0
else
    mkdir -p "${CLI_HOME_FOLDER}/bin"
    CLI_BIN_FULL_PATH="${CLI_HOME_FOLDER}/bin/${BIN_NAME}"
fi

# Move the cli from the temporary directory to the destination folder.
mv -f "${TMP_CLI}" "${CLI_BIN_FULL_PATH}"

BINARY_PRINT="${BIN_NAME}"
if [ ${PATCH_PATH} -eq 0 ]; then
    BINARY_PRINT="${CLI_BIN_FULL_PATH}"
fi

function print_install_info() {
    echo "${BIN_NAME} is installed to ${CLI_BIN_FULL_PATH}"
    echo "\$ ${BINARY_PRINT} version --full"
    "${CLI_BIN_FULL_PATH}" version --full

    if [ ! -f "${CLI_HOME_FOLDER}/config.yaml" ]; then
        echo "Please use '${BINARY_PRINT} profile create' command to create your first profile"
    fi
}

# Enable shell autocompletion.
SHELL_NAME=$(basename "${SHELL}")
case "${SHELL_NAME}" in
    bash|zsh)
        ;;
    *)
        echo "Shell command completion is not yet supported for ${SHELL_NAME}"
        print_install_info
        exit 0
        ;;
esac

function create_path_patch_file() {
    local shell_name="${1}"

    file_path="${CLI_HOME_FOLDER}/path.${shell_name}.inc"

    if [ "${shell_name}" = "bash" ]; then
        echo "cli_dir=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"" > "${file_path}"
    else
        echo "cli_dir=\"\$(cd \"\$(dirname \"\${(%):-%N}\")\" && pwd)\"" > "${file_path}"
    fi
    {
        echo "bin_path=\"\${cli_dir}/bin\""
        echo "export PATH=\"\${bin_path}:\${PATH}\""
    } >> "${file_path}"

    echo "${file_path}"
}

function patch_rc_file_for_path() {
    if [ ${PATCH_PATH} -eq 0 ]; then
        return 0
    fi

    local rc_file="${1}"
    local shell_name="${2}"

    path_patch_file=$(create_path_patch_file "${shell_name}")

    update_path_string="if [ -f '${path_patch_file}' ]; then source '${path_patch_file}'; fi"
    if ! grep -Fq "$update_path_string" "${rc_file}"; then
        # Update rc file
        if [ "${shell_name}" = "zsh" ] && [ ! -f "${rc_file}" ]; then
            echo "autoload -U compinit; compinit" > "${rc_file}"
        fi

        {
          echo ""
          echo "# The next line updates PATH for Nebius CLI."
          echo "$update_path_string"
        } >> "${rc_file}"
    fi
}

function patch_rc_file_for_autocompletion() {
    local rc_file="${1}"
    local completion_file="${2}"
    local shell_name="${3}"

    cli_completion_string="if [ -f '${completion_file}' ]; then source '${completion_file}'; fi"
    if ! grep -Fq "$cli_completion_string" "${rc_file}"; then
        {
            echo "# The next line enables shell command completion for Nebius CLI."
            echo "$cli_completion_string"
        } >> "${rc_file}"

        if [ "${shell_name}" = "bash" ]; then
            echo "${BIN_NAME} bash completion has been added to your '${rc_file}' profile."
            if [ "${OS}" = "darwin" ]; then
                echo "Make sure bash-completion (brew install bash-completion) is installed and added to your .bash_profile"
            elif [ "${OS}" == "linux" ]; then
                echo "Make sure bash-completion (apt install bash-completion) is installed."
            fi
        elif [ "${shell_name}" = "zsh" ]; then
            echo "${BIN_NAME} zsh completion has been added to your '${rc_file}' profile."
        fi
    fi
}

function create_autocompletion_file() {
    local filepath="${1}"
    local shell_name="${2}"
    "${CLI_BIN_FULL_PATH}" completion "${shell_name}" > "${filepath}"
}

function get_default_completion_file_path() {
    local shell_name="${1}"

    if [ "${shell_name}" = "bash" ]; then
        if [ "${OS}" = "darwin" ]; then
            echo "$(brew --prefix)/etc/bash_completion.d/${BIN_NAME}"
        elif [ "${OS}" = "linux" ]; then
            echo "/usr/share/bash-completion/completions/${BIN_NAME}"
        fi
    elif [ "${shell_name}" = "zsh" ]; then
        if [ "${OS}" = "darwin" ]; then
            echo "$(brew --prefix)/share/zsh/site-functions/_${BIN_NAME}"
        elif [ "${OS}" = "linux" ]; then
            echo "/usr/local/share/zsh/site-functions/_${BIN_NAME}"
        fi
    fi
}

function get_default_rc_path() {
    local shell_name="${1}"
    local os="${2}"

    default_rc_path="${HOME}/.${shell_name}rc"
    if [ "${shell_name}" == "bash" ] && [ "${os}" == "darwin" ]; then
        default_rc_path="${HOME}/.bash_profile"
    fi
    echo "$default_rc_path"
}

function shell_setting() {
    local shell_name="${1}"

    default_rc_path=$(get_default_rc_path "${shell_name}" "${OS}")
    patch_rc_file_for_path "${default_rc_path}" "${shell_name}"

    default_autocompletion_file=$(get_default_completion_file_path "${shell_name}")

    if [[ "$default_autocompletion_file" != "" ]] && [[ -w $(dirname "${default_autocompletion_file}") ]]; then
        create_autocompletion_file "${default_autocompletion_file}" "${shell_name}"
    else
        mkdir -p "${CLI_HOME_FOLDER}"
        COMPLETION_FILE="${CLI_HOME_FOLDER}/completion.${shell_name}.inc"
        create_autocompletion_file "${COMPLETION_FILE}" "${shell_name}"
        patch_rc_file_for_autocompletion "${default_rc_path}" "${COMPLETION_FILE}" "${shell_name}"
    fi
}

function full_installation() {
    shell_setting "${SHELL_NAME}"

    if [[ "${CUSTOM_INSTALL_FOLDER}" != "" ]]; then
        echo "Don't forget to add \"${CUSTOM_INSTALL_FOLDER}\" to your \$PATH"
    fi

    echo "To complete installation, start a new shell (exec -l \$SHELL)"
    print_install_info
}

if [ ! -t 0 ]; then
    # stdin is not terminal - we're piped. Skip all interactivity.
    full_installation
    exit 0
fi

function input_yes_no() {
    while read -r answer; do
        case "${answer}" in
        "Yes" | "yes" | "Y" | "y" | "")
            return 0
            ;;
        "No" | "no" | "N" | "n" )
            return 1
            ;;
        *)
            echo "Please enter 'y' or 'n': "
            ;;
        esac
    done
}

if [ ${PATCH_PATH} -eq 1 ]; then
    echo -n "Update \$PATH and enable shell command completion? [Y/n] "
else
    echo -n "Would you like to enable shell command completion? [Y/n] "
fi

if input_yes_no ; then
    full_installation
else
    COMPLETION_FILE="${CLI_HOME_FOLDER}/completion.${SHELL_NAME}.inc"
    create_autocompletion_file "${COMPLETION_FILE}" "${SHELL_NAME}"
    echo "Source '${COMPLETION_FILE}' in your profile to enable shell command completion for ${BIN_NAME}."
    if [ ${PATCH_PATH} -eq 1 ]; then
        CLI_BASH_PATH="${CLI_HOME_FOLDER}/path.${SHELL_NAME}.inc"
        echo "Source '${CLI_BASH_PATH}' in your profile to add the command line tools to your \$PATH."
    fi
    print_install_info
fi
