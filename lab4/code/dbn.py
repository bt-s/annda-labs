#!/usr/bin/python3

"""dbn.py - Containing the DeepBeliefNet class.

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology

Note: Part of this code was provided by the course coordinators.
"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


from util import *
from rbm import RestrictedBoltzmannMachine as RBM


class DeepBeliefNet():
    """A Deep Belief Net

    network          : [top] <---> [pen] ---> [hid] ---> [vis]
                               `-> [lbl]
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible"""

    def __init__(self, sizes, image_size, n_labels, batch_size):
        """Class Constructior

        Args:
            sizes (dict): Dictionary of layer names and dimensions
            image_size (list): Image dimension of data
            n_labels (int): Number of label categories
            batch_size (int): Size of mini-batch
        """

        self.rbm_stack = {
            'vis--hid': RBM(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--pen': RBM(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"],
                batch_size=batch_size),

            'pen+lbl--top': RBM(ndim_visible=sizes["pen"] + sizes["lbl"],
                ndim_hidden=sizes["top"], is_top=True, n_labels=n_labels,
                batch_size=batch_size)
        }

        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5
        self.print_period = 2000


    def recognize(self, X, y):
        """Recognize/Classify the data into label categories and calc accuracy

        Args:
          X (np.ndarray): visible data of shape (number of samples,
                          size of visible layer)
          y (np.ndarray): true labels of shape (number of samples,
                          size of label layer). Used only for calculating
                          accuracy, not driving the net
        """
        y_init = np.ones(y.shape) * 0.1 # Uninformed labels

        # Specify the vis--hid RBM
        vis__hid = self.rbm_stack["vis--hid"]

        # Specify the hid--pen RBM
        hid__pen = self.rbm_stack["hid--pen"]

        # Specify the pen+lbl--top RBM
        pen_lbl__top = self.rbm_stack["pen+lbl--top"]

        # Forward propagation through the network
        fp_bottom_h_prob, fp_bottom_h_state = vis__hid.get_h_given_v(X,
                directed=True, direction="up")
        fp_interm_h_prob, fp_interm_h_state = hid__pen.get_h_given_v(
                fp_bottom_h_prob, directed=True, direction="up")

        # Perform alternating Gibbs sampling
        v_prob = np.hstack((fp_interm_h_prob, y_init))
        for _ in range(self.n_gibbs_recog):
            h_prob, h_state = pen_lbl__top.get_h_given_v(v_prob)
            v_prob, v_state = pen_lbl__top.get_v_given_h(h_state)

        y_pred = v_state[:, -10:]

        print ("accuracy = %.2f%%" % (100. * np.mean(np.argmax(
            y_pred, axis=1) == np.argmax(y, axis=1))))


    def generate(self, X, y, name):
        """Generate data from labels

        Args:
          y (np.ndarray): true labels shaped (number of samples,
                          number of classes)
          name (str): used for saving a video of generated visible activations
        """
        n_sample, records = y.shape[0], []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        # Specify the vis--hid RBM
        vis__hid = self.rbm_stack["vis--hid"]

        # Specify the hid--pen RBM
        hid__pen = self.rbm_stack["hid--pen"]

        # Specify the pen+lbl--top RBM
        pen_lbl__top = self.rbm_stack["pen+lbl--top"]

        # Forward propagation through the network
        fp_bottom_h_prob, fp_bottom_h_state = vis__hid.get_h_given_v(X,
                directed=True, direction="up")
        fp_interm_h_prob, fp_interm_h_state = hid__pen.get_h_given_v(
                fp_bottom_h_prob, directed=True, direction="up")

        # Perform alternating Gibbs sampling
        y = np.repeat(y, X.shape[0], axis=0)
        v_state = np.hstack((fp_interm_h_state, y))

        # Perform Gibbs sampling
        for it in range(self.n_gibbs_gener):
            h_prob, h_state = pen_lbl__top.get_h_given_v(v_state)
            v_prob, v_state = pen_lbl__top.get_v_given_h(h_state)
            v_state[:, -10:] = y # fix y

            if it % 10 == 0:
                v_state_data_only = np.copy(v_state[:, :-10])

                # Backward propagation
                bp_hid_h_prob, bp_hid_h_state = hid__pen.get_v_given_h(v_state_data_only,
                        directed=True, direction="down")
                bp_vis_h_prob, bp_vis_h_state = vis__hid.get_v_given_h(bp_hid_h_state,
                        directed=True, direction="down")

                records.append([ax.imshow(np.mean(bp_vis_h_prob, axis=0).reshape(
                    self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True,
                    interpolation=None)])

            anim = stitch_video(fig,records).save("plots_and_animations/%s.generate%d.mp4" %
                    (name,np.argmax(y)))


    def train_greedylayerwise(self, X, y, n_iterations,
            load_from_file=False, save_to_file=False):
        """Greedy layer-wise training by stacking RBMs.

        Notice that once you stack more layers on top of a RBM, the weights are
        permanently untwined.

        Args:
          X (np.ndarray): Visible data shaped (size of training set,
                          size of visible layer)
          y (np.ndarray): Label data shaped (size of training set,
                          size of label layer)
          n_iterations (int): Number of iterations of learning (each iteration
                              learns a mini-batch)
          load_from_file (bool): Whether to load from file
          save_to_file (bool): Whether to save to file
        """
        if load_from_file:
            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")

        else:
            ## RBM VIS--HID
            print ("\n>> Training RBM vis--hid...")

            # Learn the weights of the vis--hid RBM by means of CD1
            self.rbm_stack["vis--hid"].cd1(X, n_iterations=n_iterations)

            # Save layer represenetations to file if requested
            if save_to_file: self.savetofile_rbm(loc="trained_rbm",
                    name="vis--hid")

            # Untwine the weights after learning
            self.rbm_stack["vis--hid"].untwine_weights()


            ## RBM HID--PEN
            print ("\n>> Training RBM hid--pen...")

            # Learn the weights of the hid--pen RBM by means of CD1
            self.rbm_stack["hid--pen"].cd1(self.rbm_stack["vis--hid"].H,
                    n_iterations=n_iterations)

            # Save layer represenetations to file if requested
            if save_to_file: self.savetofile_rbm(loc="trained_rbm",
                    name="hid--pen")

            # Untwine the weights after learning
            self.rbm_stack["hid--pen"].untwine_weights()


            ## RBM PEN+LBL--TOP
            print ("\n>> Training layer pen+lbl--top...")

            # Learn the weights of the pen+lbl--top RBM by means of CD1
            self.rbm_stack["pen+lbl--top"].cd1(np.hstack(
                (self.rbm_stack["hid--pen"].H, y)),
                n_iterations=n_iterations)

            # Save layer represenetations to file if requested
            if save_to_file: self.savetofile_rbm(loc="trained_rbm",
                    name="pen+lbl--top")


    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):
        """Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset (np.ndarray): visible data shaped (size of training set,
                                     size of visible layer)
          lbl_trainset (np.ndarray): label data shaped (size of training set,
                                     size of label layer)
          n_iterations (int): number of iterations of learning (each iteration
                              learns a mini-batch)
        """
        print ("\nTraining wake-sleep...")

        try :
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")

        except IOError :
            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):
                # wake-phase : drive the network bottom-to-top using visible
                # and label data

                # alternating Gibbs sampling in the top RBM : also store
                # neccessary information for learning this RBM

                # sleep phase : from the activities in the top RBM, drive the
                # network top-to-bottom

                # predictions : compute generative predictions from wake-phase
                # activations, and recognize predictions from sleep-phase
                # activations

                # update generative parameters :
                # here you will only use "update_generate_params" method from
                # rbm class

                # update parameters of top rbm:
                # here you will only use "update_params" method from rbm class

                # update generative parameters :
                # here you will only use "update_recognize_params" method from
                # rbm class

                if it % self.print_period == 0 : print ("iteration=%7d"%it)

            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")

        return


    def loadfromfile_rbm(self, loc, name):
        """Load RBM from file

        Args:
            loc (str): The location of the file
            name (str): Name of RBM
        """
        self.rbm_stack[name].weight_vh = np.load(f"{loc}/rbm.{name}.weight_vh.npy",
                allow_pickle=True)
        self.rbm_stack[name].bias_v    = np.load(f"{loc}/rbm.{name}.bias_v.npy",
                allow_pickle=True)
        self.rbm_stack[name].bias_h    = np.load(f"{loc}/rbm.{name}.bias_h.npy",
                allow_pickle=True)
        print(f"Loaded rbm[{name}] from {loc}.")


    def savetofile_rbm(self, loc, name):
        """Save RBM to file

        Args:
            loc (str): The location of the file
            name (str): Name of RBM
        """
        np.save(f"{loc}/rbm.{name}.weight_vh", self.rbm_stack[name].weight_vh)
        np.save(f"{loc}/rbm.{name}.bias_v", self.rbm_stack[name].bias_v)
        np.save(f"{loc}/rbm.{name}.bias_h", self.rbm_stack[name].bias_h)


    def loadfromfile_dbn(self, loc, name):
        """Load DBN from file

        Args:
            loc (str): The location of the file
            name (str): Name of RBM
        """
        self.rbm_stack[name].weight_v_to_h = np.load(f"{loc}/dbn.{name}.weight_v_to_h.npy",
                allow_pickle=True)
        self.rbm_stack[name].weight_h_to_v = np.load(f"{loc}/dbn.{name}.weight_h_to_v.npy",
                allow_pickle=True)
        self.rbm_stack[name].bias_v        = np.load(f"{loc}/dbn.{name}.bias_v.npy",
                allow_pickle=True)
        self.rbm_stack[name].bias_h        = np.load(f"{loc}/dbn.{name}.bias_h.npy")
        print(f"Loaded rbm[{name}] from {loc}.")


    def savetofile_dbn(self, loc, name):
        """Save DBN to file

        Args:
            loc (str): The location of the file
            name (str): Name of RBM
        """
        np.save(f"{loc}/dbn.{name}.weight_v_to_h", self.rbm_stack[name].weight_v_to_h)
        np.save(f"{loc}/dbn.{name}.weight_h_to_v", self.rbm_stack[name].weight_h_to_v)
        np.save(f"{loc}/dbn.{name}.bias_v",        self.rbm_stack[name].bias_v)
        np.save(f"{loc}/dbn.{name}.bias_h",        self.rbm_stack[name].bias_h)

