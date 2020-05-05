class hourglass:
    @staticmethod
    def hg_net(net, n, scope=None, reuse=None):
        num = int(net.shape[-1].value)
        sc_current = 'hg_net_{}'.format(n)
        with tf.variable_scope(scope, sc_current, [net], reuse=reuse):
            upper0 = incept_resnet.resnet_k(net)

            lower0 = slim.max_pool2d(net, 3, stride=2)
            lower0 = incept_resnet.resnet_k(lower0)

            lower0 = slim.conv2d(lower0, num * 2, 1, stride=1)

            if 1 < n:
                lower1 = hourglass.hg_net(lower0, n - 1)
            else:
                lower1 = lower0

            lower1 = slim.conv2d(lower1, num, 1, stride=1)

            lower2 = incept_resnet.resnet_k(lower1)
            upper1 = slim.conv2d_transpose(
                lower2, int(net.shape[-1].value), 3, stride=2)
            return upper0 + upper1
