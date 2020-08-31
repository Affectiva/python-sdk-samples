"""registrar.py manages identity registrations
    Usage:

    registrar register <identity> <name> <video> - Use <video> to register a face to
        the given <identity> and corresponding <name>
    registrar list - List all identities that have been registered.
    registrar unregister <identity> - unregister <identity>
    registrar unregister_all - unregister all identities

    The environment variable `AFFECTIVA_VISION_DATA_DIR` will be used get the data
    directory.
"""

import csv
from argparse import ArgumentParser
import collections
import os
import cv2
import affvisionpy
import tqdm

DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR"
NO_DATA_DIR = "To manage identities you must set " + DATA_DIR_ENV_VAR + " to a valid directory"
BAD_DATA_DIR = DATA_DIR_ENV_VAR + " is set to an invalid directory: "
ORIENTATION_TYPES = ["center", "up", "down", "left", "right"]


class FaceRegistrationStream():
    """When given a video file at `input_file_name`, registers the face found
    in that video to the identity given. For each frame, generates progress with a
    FaceRegistrationResult object that has the found face count, registration
    score and a list of face orientations still needed.
    """

    def __init__(self, input_file_name, identity, data_directory, framespan=1):
        """Initialize the generator object to read in and process
        `input_file_name`

        Parameters
        ----------
        input_file_name: string
            The name of the video file to process.

        identity: int
            The unique identity to be applied to the face in the video.

        data_directory: string
            The location of the data directory that supplies the processor with
            classifier models.

        framespan: int
            Given that it is not always necessary to capture consecutive
            frames to get an accurate registration, this will decrease the
            frequency of frame capture to speed up registration.

        Returns
        -------
        iterator
            A generator that will generate a list of dictionaries, one for each
            face found in a given frame.
        """
        self.input_file = cv2.VideoCapture(input_file_name)
        self.width = int(self.input_file.get(3))
        self.height = int(self.input_file.get(4))
        self.framespan = framespan
        point_upper_left = affvisionpy.Point(0, 0)
        point_lower_right = affvisionpy.Point(self.width, self.height)
        self.bounding_box = affvisionpy.BoundingBox(point_upper_left,
                                                    point_lower_right)
        self.identity = int(identity)

        self.face_registrar = affvisionpy.FaceRegistrar(data_directory)

        self.total_frames = int(self.input_file.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.results = None

    def __iter__(self):
        """Return's itself as an iterator."""
        return self

    def __next__(self):
        """Generate the values for the next frame, this will be a list with one
        dictionary per face found in the frame with all of the data for that
        face.

        Returns
        -------
        FaceRegistrationResult: A value with the number of faces found in the
        frame, registration score and list of face orientations still needed.
        """
        ret, frame = self.input_file.read()

        if ret and (self.frame_count % self.framespan == 0):
            timestamp = self.input_file.get(cv2.CAP_PROP_POS_MSEC)
            afframe = affvisionpy.Frame(self.width, self.height, frame,
                                        affvisionpy.ColorFormat.bgr,
                                        int(timestamp))
            result = self.face_registrar.register_face(self.identity,
                                                       afframe,
                                                       self.bounding_box)
            self.frame_count += 1
            score = result.score
            face_found = 1 if result.face_found else 0
            hints = result.orientation_hints
            orientation = affvisionpy.FaceOrientation
            self.results = collections.OrderedDict()
            self.results["time_stamp"] = int(timestamp)
            self.results["identity"] = self.identity
            self.results["face_found"] = face_found
            self.results["score"] = score
            self.results["center"] = int(orientation.center in hints)
            self.results["up"] = int(orientation.up in hints)
            self.results["down"] = int(orientation.down in hints)
            self.results["left"] = int(orientation.left in hints)
            self.results["right"] = int(orientation.right in hints)
            return self.results
        elif ret:
            self.frame_count += 1
            return self.results

        self.input_file.release()
        raise StopIteration()

    def __len__(self):
        """Returns the remaining number of un-generated frames."""
        return self.total_frames - self.frame_count


def register_id(args):
    """Register the given identity using the given video.

    Parameters
    ----------
    args: A dictionary containing:
        data_dir: The data directory.
        identity: An identity to register.
        name: The name of the person being registered.
        video: The video to register the identity with.
    """
    video = os.path.expanduser(args.video)
    assert os.path.isfile(video), "File does not exist: %s" % video
    frame_gen = FaceRegistrationStream(video, args.identity,
                                       args.data_dir, args.framespan)
    frame_progress = tqdm.tqdm(frame_gen)
    id_progress = tqdm.tqdm(total=100)

    final_score = 0
    for frame in frame_progress:
        id_progress.n = final_score = frame['score']
        id_progress.set_description("|".join([otype for otype
                                              in ORIENTATION_TYPES
                                              if frame[otype]]))

    if final_score > 0:
        add_row_to_csv(args.identity, args.name, args.identities_csv)


def list_ids(args):
    """List the identities that are currently registered.

    Parameters
    ----------
    args: A dictionary containing:
        data_dir: The data directory.
    """
    try:
        face_registrar = affvisionpy.FaceRegistrar(args.data_dir)
        identities = face_registrar.get_identities()
        print("Registered identities:")
        # if there's an identities.csv file, include the namesin the output
        if os.path.exists(args.identities_csv):
            identities_dict = read_identities_csv(args.identities_csv)
            for identity in identities:
                print(str(identity) + ": " + identities_dict[str(identity)])
        else:
            for identity in sorted(identities):
                print(identity)

    except Exception as ex:
        print(ex)


def unregister_id(args):
    """Unregister the given identity.

    Parameters
    ----------
    args: A dictionary containing:
        identity: An identity to unregister.
    """
    identity = args.identity
    face_registrar = affvisionpy.FaceRegistrar(args.data_dir)
    try:
        face_registrar.unregister(identity)
        print("Unregistered identity: " + str(identity))
    except RuntimeError as rte:
        if "unregistered identity" in rte.args[0]:
            print("%i is not registered as an identity." % identity)
        else:
            raise rte

    remove_data_from_csv(identity, args.identities_csv)


def unregister_all_ids(args):
    """Unregister all of the identities that are currently registered.

    Parameters
    ----------
    args: A dictionary containing:
        data_dir: The data directory.
    """
    face_registrar = affvisionpy.FaceRegistrar(args.data_dir)
    ids = face_registrar.get_identities()
    face_registrar.unregister_all()
    print("Unregistered identities: " + ", ".join(map(str, sorted(ids))))

    if os.path.isfile(args.identities_csv):
        os.remove(args.identities_csv)


def manage_data_dir(parser):
    """Add the data directory to the parser values, if the data directory
    isn't present or valid, return an appropriate message.

    Parameters
    ----------
    parser: the ArgumentParser
    """
    data_dir_env = os.environ.get(DATA_DIR_ENV_VAR)
    if not data_dir_env:
        parser.error(NO_DATA_DIR)
    elif not os.path.exists(os.path.expanduser(data_dir_env)):
        parser.error(BAD_DATA_DIR + data_dir_env)
    else:
        parser.set_defaults(data_dir=data_dir_env)


def manage_identities_csv(parser):
    """Assemble the path to the identities.csv file

    Parameters
    ----------
    parser: the ArgumentParser
    """
    data_dir_env = os.environ.get(DATA_DIR_ENV_VAR)
    identities_csv_path =  data_dir_env + '/attribs/identities.csv'
    parser.set_defaults(identities_csv=identities_csv_path)


def add_row_to_csv(identity, name, identities_csv):
    """Add a row to the identities.csv file with the specified identity and name.  If there's already a row
    with the specified identity, no change will be made

    Parameters
    ----------
    identity: numeric identity of the registered person
    name: name of the registered person
    identities_csv: path to the identities.csv file

    """
    with open(identities_csv, 'a', newline='') as file:
        # if empty, add header row
        if file.tell() == 0:
            file.write("identity,name\n")
        # if there's not already a row for this identity, add it.  Note that it's OK to register multiple times for
        # the same identity (e.g. using different videos).
        if str(identity) not in read_identities_csv(identities_csv):
            file.write(str(identity) + "," + name + "\n")


def read_identities_csv(identities_csv):
    """Read the identities.csv file and return its contents (minus the header row) as a dict

    Parameters
    ----------
    identities_csv: path to the identities.csv file
    """
    lines = {}
    with open(identities_csv, 'r') as input_file:
        reader = csv.reader(input_file)
        next(reader, None)  # skip header row
        for row in reader:
            lines[row[0]] = row[1]
    return lines


def remove_data_from_csv(identity, identities_csv):
    """If the identities.csv file exists, remove the row that corresponds to the specified identity

    Parameters
    ----------
    identity: numeric identity of the registered person
    identities_csv: path to the identities.csv file
    """
    if not os.path.isfile(identities_csv):
        return

    # read the file into a dict and remove the element for this identity
    identities_dict = read_identities_csv(identities_csv)
    if str(identity) in identities_dict.keys():
        identities_dict.pop(str(identity))

    # rewrite the file
    with open(identities_csv, 'w') as file:
        file.write("identity,name\n")
        for key in identities_dict.keys():
            file.write("%s,%s\n" % (key, identities_dict[key]))


def process_arguments():
    """Read and parse the arguments from the commandline. Executes the correct
    command for each of the available commands: register, list, unregister,
    unregister_all
    """
    description = "Utility for managing face registration."
    epilog = ("The environment variable " + DATA_DIR_ENV_VAR + " must be set "
              "to the Affectiva SDK's data directory to use this script.")
    parser = ArgumentParser(description=description, epilog=epilog)
    subparsers = parser.add_subparsers()
    manage_data_dir(parser)
    manage_identities_csv(parser)

    # Register identity
    reg = subparsers.add_parser("register", help="Register an identity")
    reg.add_argument("identity", type=int, help="The numeric identity of the person being registered")
    reg.add_argument("name", help="The name of the person being registered")
    reg.add_argument("video", help="A video file containing the face to register")
    reg.add_argument("-f", "--framespan", type=int, default=1, help="Only use every nth frame from the video")
    reg.set_defaults(func=register_id)

    # List identities
    lst = subparsers.add_parser("list", help="List registered identities")
    lst.set_defaults(func=list_ids)

    # Unregister identity
    unregister = subparsers.add_parser("unregister", help="Unregister an identity")
    unregister.add_argument("identity", type=int, help="The identity to unregister")
    unregister.set_defaults(func=unregister_id)

    # Unregister all identities
    unregister_all = subparsers.add_parser("unregister_all", help="Unregister all identities")
    unregister_all.set_defaults(func=unregister_all_ids)

    return parser


if __name__ == "__main__":
    PARSER = process_arguments()
    ARGS = PARSER.parse_args()

    try:
        ARGS.func(ARGS)
    except AttributeError:
        PARSER.print_help()
